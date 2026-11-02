#!/usr/bin/env python3
"""Sync Home Assistant configuration over the HTTP API (no SSH needed).

Works through any URL that reaches Home Assistant's frontend, including a
Nabu Casa remote URL, so it can be used where SSH/rsync is not available.

Commands:
  pull  Download UI-managed automations, scripts and scenes into
        config/automations.yaml, config/scripts.yaml and config/scenes.yaml,
        and the entity/device/area registries into config/.storage/ so the
        reference validator and entity explorer work.
  push  Upload local automations, scripts and scenes that differ from Home
        Assistant through the config API. Home Assistant writes its own YAML
        files and reloads the component itself. Items that exist only in Home
        Assistant are reported, and deleted only with --delete.

Automations and scripts defined in packages or other included files are
written to config/readonly/ for reference; they cannot be pushed.

Limitations: configuration.yaml, packages, blueprints and other files are not
reachable through the API. Automations and scenes need an ``id`` to be managed.

Requires HA_URL and HA_TOKEN (environment or .env). The token must belong to
an administrator.
"""

import argparse
import asyncio
import json
import os
import ssl
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import yaml

CONFIG_DIR = Path("config")
# Written by pull; push refuses to run without it so the repository's example
# files are never uploaded over a real configuration.
PULL_MARKER = CONFIG_DIR / ".storage" / "api_sync_pulled_from"
# Automations/scripts defined outside the UI-managed files (packages, includes).
# The API can read them but not write them, so they are kept apart for reference.
READONLY_DIR = CONFIG_DIR / "readonly"
# The real configuration.yaml cannot be read through the API. Pull writes this
# stub so local validation loads the pulled files instead of repository examples.
CONFIGURATION_STUB = """\
# STUB written by tools/ha_api_sync.py pull. The real configuration.yaml on
# Home Assistant cannot be read through the API, and this file is never pushed.
# It only lets local validation load the UI-managed files.
automation: !include automations.yaml
script: !include scripts.yaml
scene: !include scenes.yaml
"""
READONLY_HEADER = (
    "# READ-ONLY reference pulled by tools/ha_api_sync.py.\n"
    "# These are defined in packages or other files on Home Assistant and cannot\n"
    "# be changed through the API. Edits here are never pushed.\n"
)
TIMEOUT = 30

# kind -> (yaml file, config API path, entity domain)
KINDS = {
    "automation": ("automations.yaml", "automation", "automation"),
    "script": ("scripts.yaml", "script", "script"),
    "scene": ("scenes.yaml", "scene", "scene"),
}


def load_env_file() -> None:
    """Load variables from .env without overriding the real environment."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class HAClient:
    """Minimal Home Assistant REST + WebSocket client."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )

    def get(self, path: str) -> requests.Response:
        return self.session.get(f"{self.url}{path}", timeout=TIMEOUT)

    def post(self, path: str, data: Any) -> requests.Response:
        return self.session.post(f"{self.url}{path}", json=data, timeout=TIMEOUT)

    def delete(self, path: str) -> requests.Response:
        return self.session.delete(f"{self.url}{path}", timeout=TIMEOUT)

    def states(self) -> List[Dict[str, Any]]:
        resp = self.get("/api/states")
        resp.raise_for_status()
        return resp.json()

    def get_item_config(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        resp = self.get(f"/api/config/{KINDS[kind][1]}/config/{quote(key, safe='')}")
        if resp.status_code in (400, 404):
            return None  # Not managed through the UI YAML file
        resp.raise_for_status()
        return resp.json()

    def save_item_config(self, kind: str, key: str, config: Dict[str, Any]) -> None:
        resp = self.post(
            f"/api/config/{KINDS[kind][1]}/config/{quote(key, safe='')}", config
        )
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code}: {resp.text.strip()}")

    def delete_item_config(self, kind: str, key: str) -> None:
        resp = self.delete(f"/api/config/{KINDS[kind][1]}/config/{quote(key, safe='')}")
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code}: {resp.text.strip()}")

    def ws_commands(
        self, commands: List[Dict[str, Any]], allow_errors: bool = False
    ) -> List[Any]:
        """Run WebSocket commands and return their results in order.

        With allow_errors, a failed command gives None instead of raising.
        """
        return asyncio.run(self._ws_commands(commands, allow_errors))

    async def _ws_commands(
        self, commands: List[Dict[str, Any]], allow_errors: bool
    ) -> List[Any]:
        try:
            from websockets.asyncio.client import connect
        except ImportError as err:  # pragma: no cover
            raise RuntimeError("Install 'websockets' (pip install websockets)") from err

        ws_url = "ws" + self.url[4:] + "/api/websocket"
        ssl_ctx = None
        if ws_url.startswith("wss://"):
            ssl_ctx = ssl.create_default_context(
                cafile=os.environ.get("SSL_CERT_FILE")
                or os.environ.get("REQUESTS_CA_BUNDLE")
            )
        async with connect(ws_url, ssl=ssl_ctx, max_size=None) as ws:
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_required":
                raise RuntimeError(f"Unexpected WebSocket greeting: {msg}")
            await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_ok":
                raise RuntimeError(f"WebSocket authentication failed: {msg}")
            results = []
            for idx, command in enumerate(commands, start=1):
                await ws.send(json.dumps({"id": idx, **command}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == idx and msg.get("type") == "result":
                        break
                if not msg.get("success"):
                    if allow_errors:
                        results.append(None)
                        continue
                    raise RuntimeError(f"{command['type']} failed: {msg.get('error')}")
                results.append(msg.get("result"))
            return results


# ---------------------------------------------------------------- YAML helpers


class _Dumper(yaml.SafeDumper):
    """YAML dumper that keeps key order and indents lists like HA does."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _str_presenter(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_presenter)


def dump_yaml(data: Any) -> str:
    if not data:
        return "[]\n" if isinstance(data, list) else "{}\n"
    return yaml.dump(
        data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=1000
    )


class _Loader(yaml.SafeLoader):
    """Safe loader that keeps HA tags (e.g. !secret, !input) as plain text."""


def _tag_constructor(loader, tag_suffix, node):
    return f"{tag_suffix} {loader.construct_scalar(node)}"


_Loader.add_multi_constructor("!", _tag_constructor)


def load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    return yaml.load(path.read_text(), Loader=_Loader)


# ------------------------------------------------------------- remote reading


def remote_keys(states: List[Dict[str, Any]], kind: str) -> Dict[str, str]:
    """Return {config key: entity_id} for items of a kind."""
    domain = KINDS[kind][2]
    keys = {}
    for state in states:
        entity_id = state["entity_id"]
        if not entity_id.startswith(domain + "."):
            continue
        if state["attributes"].get("restored"):
            continue  # Orphaned registry entry; its config no longer exists
        if kind == "script":
            keys[entity_id.split(".", 1)[1]] = entity_id
        elif state["attributes"].get("id"):
            keys[str(state["attributes"]["id"])] = entity_id
    return keys


def fetch_remote(
    client: HAClient, states: List[Dict[str, Any]], kind: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Return {key: config} for UI-managed items and {key: entity_id} of others."""
    configs, skipped = {}, {}
    for key, entity_id in remote_keys(states, kind).items():
        config = client.get_item_config(kind, key)
        if config is None:
            skipped[key] = entity_id
        else:
            configs[key] = config
    return configs, skipped


def local_items(kind: str) -> Dict[str, Dict[str, Any]]:
    """Return {key: config} from the local YAML file."""
    data = load_yaml(CONFIG_DIR / KINDS[kind][0])
    if not data:
        return {}
    if kind == "script":
        if not isinstance(data, dict):
            raise ValueError("scripts.yaml must be a mapping")
        return {str(key): value for key, value in data.items()}
    if not isinstance(data, list):
        raise ValueError(f"{KINDS[kind][0]} must be a list")
    items = {}
    for item in data:
        if "id" not in item:
            name = item.get("alias") or item.get("name")
            raise ValueError(f"{KINDS[kind][0]}: item without id: {name!r}")
        items[str(item["id"])] = item
    return items


# ----------------------------------------------------------------------- pull


def sort_items(kind: str, configs: Dict[str, Dict[str, Any]]) -> Any:
    if kind == "script":
        return dict(sorted(configs.items()))
    return sorted(
        configs.values(),
        key=lambda c: str(c.get("alias") or c.get("name") or c.get("id")),
    )


def pull_items(client: HAClient, states: List[Dict[str, Any]]) -> None:
    READONLY_DIR.mkdir(parents=True, exist_ok=True)
    for kind, (filename, _, _) in KINDS.items():
        configs, skipped = fetch_remote(client, states, kind)
        (CONFIG_DIR / filename).write_text(dump_yaml(sort_items(kind, configs)))
        print(f"✅ {filename}: {len(configs)} UI-managed {kind}s")

        readonly_path = READONLY_DIR / filename
        if kind == "scene" or not skipped:
            readonly_path.unlink(missing_ok=True)
            continue
        # Items defined in packages or other files: readable over WebSocket only.
        results = client.ws_commands(
            [{"type": f"{kind}/config", "entity_id": eid} for eid in skipped.values()],
            allow_errors=True,
        )
        others = {k: r["config"] for k, r in zip(skipped, results) if r}
        unreadable = [eid for eid, r in zip(skipped.values(), results) if not r]
        if unreadable:
            print(f"⚠️  could not read config of: {', '.join(unreadable)}")
        readonly_path.write_text(READONLY_HEADER + dump_yaml(sort_items(kind, others)))
        print(f"📖 readonly/{filename}: {len(others)} {kind}s defined in other files")


def pull_registries(client: HAClient, states: List[Dict[str, Any]]) -> None:
    entities, devices, areas = client.ws_commands(
        [
            {"type": "config/entity_registry/list"},
            {"type": "config/device_registry/list"},
            {"type": "config/area_registry/list"},
        ]
    )
    # The list command leaves out device class and unit; take them from states.
    attrs = {s["entity_id"]: s["attributes"] for s in states}
    for entity in entities:
        state_attrs = attrs.get(entity["entity_id"], {})
        entity.setdefault("original_device_class", state_attrs.get("device_class"))
        entity.setdefault("device_class", None)
        entity.setdefault("unit_of_measurement", state_attrs.get("unit_of_measurement"))
    # Entities without a unique_id (e.g. YAML-defined) exist only as states.
    registered = {e["entity_id"] for e in entities}
    for entity_id, state_attrs in attrs.items():
        if entity_id not in registered:
            entities.append(
                {
                    "entity_id": entity_id,
                    "platform": entity_id.split(".", 1)[0],
                    "name": None,
                    "original_name": state_attrs.get("friendly_name"),
                    "original_device_class": state_attrs.get("device_class"),
                    "unit_of_measurement": state_attrs.get("unit_of_measurement"),
                    "disabled_by": None,
                    "hidden_by": None,
                    "states_only": True,
                }
            )
    for area in areas:
        area.setdefault("id", area.get("area_id"))

    storage = CONFIG_DIR / ".storage"
    storage.mkdir(parents=True, exist_ok=True)
    for key, field, items in (
        ("core.entity_registry", "entities", entities),
        ("core.device_registry", "devices", devices),
        ("core.area_registry", "areas", areas),
    ):
        payload = {"version": 1, "minor_version": 1, "key": key, "data": {field: items}}
        (storage / key).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"✅ .storage/{key}: {len(items)} {field}")


def cmd_pull(client: HAClient, args: argparse.Namespace) -> int:
    states = client.states()
    if not args.registries_only:
        pull_items(client, states)
        (CONFIG_DIR / "configuration.yaml").write_text(CONFIGURATION_STUB)
    pull_registries(client, states)
    if not args.registries_only:
        PULL_MARKER.write_text(client.url + "\n")
    print(
        "ℹ️  configuration.yaml is a local stub; the real one is not readable via API"
    )
    return 0


# ----------------------------------------------------------------------- push


def normalize(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)


def cmd_push(client: HAClient, args: argparse.Namespace) -> int:
    pulled_from = PULL_MARKER.read_text().strip() if PULL_MARKER.exists() else None
    if pulled_from != client.url:
        print(
            f"❌ Run 'pull' from {client.url} first; local files may not be its config"
        )
        return 1
    states = client.states()
    failures = 0
    changes = 0
    for kind in KINDS:
        local = local_items(kind)
        remote, skipped = fetch_remote(client, states, kind)

        for key, config in local.items():
            if key in skipped:
                print(f"⚠️  {kind} {key}: defined outside the UI YAML file, not pushed")
                continue
            if key in remote and normalize(remote[key]) == normalize(config):
                continue
            action = "update" if key in remote else "create"
            label = config.get("alias") or config.get("name") or key
            changes += 1
            if args.dry_run:
                print(f"📝 would {action} {kind} {key} ({label})")
                continue
            try:
                client.save_item_config(kind, key, config)
                print(f"✅ {action}d {kind} {key} ({label})")
            except RuntimeError as err:
                failures += 1
                print(f"❌ {kind} {key}: {err}")

        for key in sorted(set(remote) - set(local)):
            label = remote[key].get("alias") or remote[key].get("name") or key
            if not args.delete:
                print(f"ℹ️  {kind} {key} ({label}) exists only in HA (use --delete)")
                continue
            changes += 1
            if args.dry_run:
                print(f"📝 would delete {kind} {key} ({label})")
                continue
            try:
                client.delete_item_config(kind, key)
                print(f"🗑️  deleted {kind} {key} ({label})")
            except RuntimeError as err:
                failures += 1
                print(f"❌ {kind} {key}: {err}")

    if not changes:
        print("✅ Home Assistant already matches local files")
    elif args.dry_run:
        print(f"Dry run: {changes} change(s), nothing sent")
    return 1 if failures else 0


def cmd_check(client: HAClient, args: argparse.Namespace) -> int:
    resp = client.get("/api/config")
    resp.raise_for_status()
    info = resp.json()
    print(f"✅ REST API: {info.get('location_name')} {info.get('version')}")
    (user,) = client.ws_commands([{"type": "auth/current_user"}])
    admin = "admin" if user.get("is_admin") else "NOT admin"
    print(f"✅ WebSocket API: user {user.get('name')} ({admin})")
    return 0 if user.get("is_admin") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("pull", help="Download config and registries from HA")
    pull.add_argument(
        "--registries-only", action="store_true", help="Only update config/.storage"
    )
    push = sub.add_parser("push", help="Upload changed automations/scripts/scenes")
    push.add_argument("--dry-run", action="store_true", help="Only show changes")
    push.add_argument(
        "--delete", action="store_true", help="Delete items missing locally"
    )
    sub.add_parser("check", help="Test REST and WebSocket access")
    args = parser.parse_args()

    load_env_file()
    url, token = os.getenv("HA_URL"), os.getenv("HA_TOKEN")
    if not url or not token:
        print("❌ HA_URL and HA_TOKEN must be set (environment or .env)")
        return 1

    client = HAClient(url, token)
    commands = {"pull": cmd_pull, "push": cmd_push, "check": cmd_check}
    try:
        return commands[args.command](client, args)
    except (requests.RequestException, RuntimeError, ValueError) as err:
        print(f"❌ {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
