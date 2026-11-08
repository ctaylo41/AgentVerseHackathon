"""Real-Time WebSocket Bridge for the Antigravity Robotic Director System.

This module implements an asynchronous WebSocket server (`AntigravityDirectorBridge`)
that connects external web dashboards, mobile UIs, or Gemma 4 agent frontends
with the real-time MuJoCo physics and Mink IK control loop in
`robot_path_planner.py`.

Communication Protocol (JSON over WebSocket, default `ws://127.0.0.1:8765`):
  - Incoming Client Commands:
    1. `{"action": "prompt", "prompt": "<natural language request>"}`:
       Parses the prompt via `GemmaDirectorParser`, updates the active robot
       shot primitive and digital twin choreography, and returns a
       `prompt_parsed` acknowledgment.
    2. `{"action": "set_shot", "shot_type": "orbit" | "dolly_in" | ...}`:
       Directly switches the active shot primitive and returns a `shot_updated`
       acknowledgment.
    3. `{"action": "get_status"}`:
       Returns an immediate snapshot of the active shot, creator instruction,
       3D positions, and latest coaching evaluation report.
  - Outgoing Server Broadcasts (10 Hz):
    Streams `{"type": "telemetry", ...}` packets containing live `user_pos`,
    `cam_pos`, `active_shot`, `instruction`, and `report` (ideal vs. actual
    deviation score and coaching tips) to all connected clients.

Threading Model:
  When launched alongside MuJoCo via `start_bridge_background()`, the asyncio
  event loop runs in a dedicated daemon thread while sharing state safely with
  the 200 Hz physics thread through a `threading.Lock`.
"""

import asyncio
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Optional, Sequence, Set, Union
import numpy as np
import websockets

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from gemma_parser import GemmaDirectorParser
from user_director_twin import UserDirectorDigitalTwin

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class AntigravityDirectorBridge:
    """Bi-directional WebSocket server bridging Gemma 4 UIs and robot simulation.

    Attributes:
        host: Hostname or IP address on which the WebSocket server listens.
        port: TCP port number on which the WebSocket server listens.
        parser: `GemmaDirectorParser` instance for natural-language shot parsing.
        coach: `UserDirectorDigitalTwin` instance for evaluating creator motion.
        connected_clients: Set of active WebSocket client connections.
        current_shot: Thread-safe string identifier of the currently active shot.
        creator_instruction: Thread-safe stage instruction for the active shot.
        user_pos: Latest 3D user position `[X, Y, Z]` in meters.
        cam_pos: Latest 3D camera target position `[X, Y, Z]` in meters.
        last_report: Latest dictionary returned by `coach.evaluate_user_performance`.
        lock: Mutex protecting shared state between the physics and asyncio threads.
        running: Boolean flag controlling the server broadcast loop lifecycle.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """Initializes the WebSocket bridge state and helper modules.

        Args:
            host: Bind address for the WebSocket server. Defaults to `'127.0.0.1'`.
            port: Bind port for the WebSocket server. Defaults to `8765`.
        """
        self.host: str = host
        self.port: int = port
        self.parser: GemmaDirectorParser = GemmaDirectorParser()
        self.coach: UserDirectorDigitalTwin = UserDirectorDigitalTwin(
            shot_type="orbit"
        )
        self.connected_clients: Set[Any] = set()

        # Shared state synchronized between the WebSocket thread and physics loop.
        self.current_shot: str = "orbit"
        self.creator_instruction: str = self.coach.instruction
        self.user_pos: np.ndarray = np.array([1.0, 0.0, 0.9])
        self.cam_pos: np.ndarray = np.array([0.45, 0.0, 0.65])
        self.last_report: Dict[str, Any] = {}
        self.lock: threading.Lock = threading.Lock()
        self.running: bool = True
        self.server: Optional[Any] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def update_telemetry(
        self,
        user_pos: Union[Sequence[float], np.ndarray],
        cam_pos: Union[Sequence[float], np.ndarray],
        t: float,
    ) -> None:
        """Pushes the latest simulation coordinates from the physics loop.

        Thread-safe method called by `RobotPathPlanner.execute_path_plan` on
        every simulation step to update telemetry and recompute digital twin
        coaching feedback.

        Args:
            user_pos: 3D position `[X, Y, Z]` of the tracked human subject (m).
            cam_pos: 3D position `[X, Y, Z]` of the robot camera end-effector (m).
            t: Elapsed simulation time in seconds.
        """
        with self.lock:
            self.user_pos = np.asarray(user_pos, dtype=float)
            self.cam_pos = np.asarray(cam_pos, dtype=float)
            self.last_report = self.coach.evaluate_user_performance(
                self.user_pos, t
            )

    def set_shot(
        self, shot_type: str, instruction: Optional[str] = None
    ) -> None:
        """Switches the active shot primitive and updates the digital twin coach.

        Args:
            shot_type: New shot identifier (`'orbit'`, `'dolly_in'`,
                `'low_angle_hero'`, or `'tracking'`).
            instruction: Optional custom creator instruction generated by Gemma 4.
                If `None`, uses the default instruction from `UserDirectorDigitalTwin`.
        """
        with self.lock:
            self.current_shot = shot_type
            self.coach.set_shot_type(shot_type)
            if instruction:
                self.creator_instruction = instruction
            else:
                self.creator_instruction = self.coach.instruction
            print(f"[Bridge] Switched shot to: {shot_type.upper()}")
            print(f"[Bridge] Instruction: {self.creator_instruction}")

    async def handle_client(self, websocket: Any) -> None:
        """Handles the lifecycle and incoming JSON commands of a WebSocket client.

        Args:
            websocket: The connected `websockets` client connection object.
        """
        self.connected_clients.add(websocket)
        print(f"[Bridge] New client connected from {websocket.remote_address}")

        try:
            # Send initial handshake state upon client connection.
            init_payload = {
                "type": "init",
                "active_shot": self.current_shot,
                "instruction": self.creator_instruction,
                "supported_shots": GemmaDirectorParser.SUPPORTED_SHOTS,
            }
            await websocket.send(json.dumps(init_payload))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get("action")

                    if action == "set_shot":
                        shot = data.get("shot_type", "tracking")
                        self.set_shot(shot)
                        ack = {
                            "type": "shot_updated",
                            "shot_type": self.current_shot,
                            "instruction": self.creator_instruction,
                        }
                        await websocket.send(json.dumps(ack))

                    elif action == "prompt":
                        # Parse free-form natural language prompt via Gemma 4.
                        prompt_text = data.get("prompt", "")
                        plan = self.parser.parse_prompt(prompt_text)
                        self.set_shot(
                            plan["shot_type"], plan.get("creator_instruction")
                        )
                        resp = {
                            "type": "prompt_parsed",
                            "parsed_plan": plan,
                            "active_shot": self.current_shot,
                            "instruction": self.creator_instruction,
                        }
                        await websocket.send(json.dumps(resp))

                    elif action == "get_status":
                        with self.lock:
                            status_payload = {
                                "type": "status",
                                "active_shot": self.current_shot,
                                "instruction": self.creator_instruction,
                                "user_position": self.user_pos.tolist(),
                                "cam_position": self.cam_pos.tolist(),
                                "coaching_report": self.last_report,
                            }
                        await websocket.send(json.dumps(status_payload))

                except Exception as exc:
                    await websocket.send(
                        json.dumps({"type": "error", "message": str(exc)})
                    )

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
            print("[Bridge] Client disconnected.")

    async def broadcast_telemetry_loop(self) -> None:
        """Broadcasts live telemetry and coaching reports to all clients at 10 Hz."""
        while self.running:
            if self.connected_clients:
                with self.lock:
                    payload = json.dumps(
                        {
                            "type": "telemetry",
                            "timestamp": time.time(),
                            "active_shot": self.current_shot,
                            "instruction": self.creator_instruction,
                            "user_pos": self.user_pos.tolist(),
                            "cam_pos": self.cam_pos.tolist(),
                            "report": self.last_report,
                        }
                    )
                websockets.broadcast(self.connected_clients, payload)
            await asyncio.sleep(0.1)

    async def run_server(self) -> None:
        """Starts the asynchronous WebSocket server and runs the broadcast loop."""
        self.loop = asyncio.get_running_loop()
        print(
            f"[Bridge] Starting WebSocket server on ws://{self.host}:{self.port}..."
        )
        async with websockets.serve(
            self.handle_client, self.host, self.port
        ) as server:
            self.server = server
            try:
                await self.broadcast_telemetry_loop()
            except asyncio.CancelledError:
                pass

    def stop(self) -> None:
        """Cleanly stops the telemetry broadcast loop and closes the server."""
        self.running = False
        if self.loop and self.server:
            self.loop.call_soon_threadsafe(self.server.close)


def start_bridge_background(
    host: str = "127.0.0.1", port: int = 8765
) -> AntigravityDirectorBridge:
    """Launches the WebSocket bridge in a non-blocking background daemon thread.

    Args:
        host: Host address to bind the server to. Defaults to `'127.0.0.1'`.
        port: Port number to bind the server to. Defaults to `8765`.

    Returns:
        The running `AntigravityDirectorBridge` instance linked to the background
        event loop.
    """
    bridge = AntigravityDirectorBridge(host, port)

    def thread_worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(bridge.run_server())
        except Exception:
            pass
        finally:
            loop.close()

    thread = threading.Thread(target=thread_worker, daemon=True)
    thread.start()
    return bridge


if __name__ == "__main__":
    # Run the WebSocket bridge as a standalone server process.
    standalone_bridge = AntigravityDirectorBridge()
    try:
        asyncio.run(standalone_bridge.run_server())
    except KeyboardInterrupt:
        print("\n[Bridge] Server stopped.")
