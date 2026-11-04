import asyncio
import json
import logging
import socket
import unittest

from plugins.virtRTG.device_link.device_adapter import DeviceAdapter, DeviceState, ProtocolError, projection_geometry


def message(kind, seq, **payload):
    return {"protocol": "virtualrtg.device", "version": 1, "type": kind,
            "seq": seq, "device_id": "browser-simulator-01",
            "timestamp": "2026-11-04T00:00:00.000Z", **payload}


def pose(p, r=(0, 0, 0)):
    return {"position_mm": list(p), "rotation_deg": list(r), "rotation_order": "XYZ"}


def state_update(seq=2, revision=1, preset="table"):
    poses = {
        "tube": pose((0, 2350, 0)), "detector": pose((0, 770, 0)),
        "phantom": pose((0, 1100, 0)), "table": pose((0, 835, 0)),
    }
    if preset == "lateral":
        poses.update(tube=pose((1100,1140,0),(0,0,-90)), detector=pose((-820,1140,0),(0,0,-90)))
    if preset == "upright":
        poses.update(phantom=pose((0,1050,0),(90,0,0)), tube=pose((0,1120,1150),(90,0,0)), detector=pose((0,1120,-680),(90,0,0)))
    return message("state.update",seq,revision=revision,device={
        "generator":{"kvp":80,"ma":200,"exposure_ms":20,"mas":4},
        "collimator":{"width_cm":24,"height_cm":30,"light_enabled":True},"poses":poses})


class AdapterTests(unittest.TestCase):
    def setUp(self): self.adapter=DeviceAdapter()

    def test_browser_initial_sync_and_round_trip(self):
        self.adapter.handle(message("device.hello",1,capabilities=["pose","generator","collimation","acquisition"],transport="mock"))
        self.adapter.handle(state_update())
        s=self.adapter.state
        self.assertEqual(s.poses["tube"].position_m,(0,2.35,0))
        self.assertEqual((s.generator.kvp,s.generator.mas),(80,4))
        rebuilt=DeviceState.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(rebuilt.to_dict(),s.to_dict())

    def test_individual_pose_update_and_units(self):
        self.adapter.handle(state_update())
        self.adapter.handle(message("device.pose",3,object="tube",revision=2,pose=pose((100,2000,-300),(90,0,0))))
        p=self.adapter.state.poses["tube"]
        self.assertEqual(p.position_m,(.1,2,-.3)); self.assertAlmostEqual(p.rotation_rad[0],3.141592653589793/2)

    def test_generator_collimator_validation(self):
        self.adapter.handle(state_update())
        msg=state_update(seq=3,revision=2); msg["device"]["generator"].update(kvp=120,ma=320,exposure_ms=50,mas=16)
        msg["device"]["collimator"].update(width_cm=18,height_cm=20,light_enabled=False)
        self.adapter.handle(msg)
        self.assertEqual(self.adapter.state.generator.mas,16); self.assertFalse(self.adapter.state.collimator.light_enabled)

    def test_preset_geometry_directions(self):
        for preset, expected in (("table",-1),("lateral",-1),("upright",-1)):
            adapter=DeviceAdapter(); adapter.handle(state_update(preset=preset))
            g=projection_geometry(adapter.state)
            ray=tuple(g["detector_active_center_m"][i]-g["tube_focal_point_m"][i] for i in range(3))
            axis=1 if preset=="table" else 0 if preset=="lateral" else 2
            self.assertLess(ray[axis],0)
            self.assertAlmostEqual(g["source_to_detector_m"],sum(x*x for x in ray)**.5)
            self.assertAlmostEqual(sum(g["detector_normal"][i]*ray[i] for i in range(3)),-sum(x*x for x in ray)**.5,places=6)

    def test_bad_payload_stale_update_and_acquisition_lifecycle(self):
        self.adapter.handle(state_update())
        before=self.adapter.state.to_dict()
        bad=state_update(seq=3,revision=2); bad["device"]["poses"]["tube"]["rotation_order"]="ZYX"
        with self.assertRaises(ProtocolError): self.adapter.handle(bad)
        self.assertEqual(self.adapter.state.to_dict(),before)
        with self.assertRaises(ProtocolError): self.adapter.handle(state_update(seq=4,revision=1))
        self.adapter.handle(message("acquisition.prepare",4,revision=1))
        self.adapter.handle(message("acquisition.request",5,request_id="demo-123",scene_revision=1,
            parameters={"kvp":80,"ma":200,"exposure_ms":20,"mas":4,"projection":"cone"}))
        self.assertEqual(self.adapter.state.acquisition.status,"requested")
        self.assertFalse(self.adapter.snapshot()["image_generated"])
        self.adapter.handle(message("acquisition.cancel",6))
        self.assertEqual(self.adapter.state.acquisition.status,"cancelled")

    def test_preset_and_pose_messages(self):
        self.adapter.handle(state_update())
        self.adapter.handle(message("procedure.select",3,preset="upright"))
        self.assertEqual(self.adapter.state.procedure,"upright")


class WebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_browser_messages_end_to_end(self):
        try: import websockets
        except ImportError: self.skipTest("websockets package unavailable")
        from websockets.asyncio.server import serve
        from plugins.virtRTG.device_link.server import websocket_handler
        adapter=DeviceAdapter()
        async with serve(lambda ws: websocket_handler(ws,adapter),"127.0.0.1",0) as server:
            port=server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps(message("device.hello",1,capabilities=["pose","generator","collimation","acquisition"])))
                ack=json.loads(await ws.recv()); self.assertEqual(ack["status"],"accepted")
                await ws.send(json.dumps(state_update()))
                ack=json.loads(await ws.recv()); self.assertEqual(ack["in_reply_to"],2)
                self.assertEqual(adapter.state.poses["detector"].position_m[1],.77)

    async def test_reconnect_starts_fresh_sequence_and_stale_seq_is_rejected(self):
        import websockets
        from websockets.asyncio.server import serve
        from plugins.virtRTG.device_link.server import websocket_handler
        adapters=[]
        async def handler(ws):
            adapter=DeviceAdapter(); adapters.append(adapter)
            await websocket_handler(ws,adapter)
        async with serve(handler,"127.0.0.1",0) as server:
            port=server.sockets[0].getsockname()[1]
            for _ in range(2):
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps(message("device.hello",1,capabilities=[])))
                    self.assertEqual(json.loads(await ws.recv())["status"],"accepted")
                    await ws.send(json.dumps(state_update(seq=2)))
                    self.assertEqual(json.loads(await ws.recv())["status"],"accepted")
                    before=adapters[-1].state.to_dict()
                    await ws.send(json.dumps(message("device.pose",2,object="tube",revision=2,pose=pose((0,1000,0)))))
                    rejected=json.loads(await ws.recv())
                    self.assertEqual(rejected["status"],"rejected")
                    self.assertEqual(adapters[-1].state.to_dict(),before)
        self.assertEqual(len(adapters),2)
        self.assertEqual([a.state.last_sequence for a in adapters],[2,2])

    async def test_logging_normal_verbose_and_rejection(self):
        import websockets
        from websockets.asyncio.server import serve
        from plugins.virtRTG.device_link.server import websocket_handler
        for verbose in (False,True):
            with self.assertLogs("virtualrtg.adapter",level="DEBUG" if verbose else "INFO") as captured:
                async with serve(lambda ws: websocket_handler(ws,DeviceAdapter(),verbose),"127.0.0.1",0) as server:
                    port=server.sockets[0].getsockname()[1]
                    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                        await ws.send(json.dumps(state_update()))
                        self.assertEqual(json.loads(await ws.recv())["status"],"accepted")
                        await ws.send(json.dumps(message("device.pose",3,object="tube",revision=2,pose=pose((25,2000,0),(0,0,5)))))
                        self.assertEqual(json.loads(await ws.recv())["status"],"accepted")
                        await ws.send(json.dumps(message("device.pose",3,object="tube",revision=3,pose=pose((99,2000,0)))))
                        self.assertEqual(json.loads(await ws.recv())["status"],"rejected")
            output="\n".join(captured.output)
            self.assertIn("Projection geometry:",output)
            self.assertIn("Pose tube:",output)
            self.assertIn("Rejected type=device.pose seq=3",output)
            self.assertEqual("Incoming JSON:" in output,verbose)
            if verbose:
                self.assertIn("detector_normal",output)
                self.assertIn("VirtualXRay mapped geometry",output)
            else:
                self.assertNotIn("VirtualXRay mapped geometry",output)

    async def test_zero_sid_and_diagnostic_failure_still_acknowledge_verbose_messages(self):
        import websockets
        from unittest.mock import patch
        from websockets.asyncio.server import serve
        from plugins.virtRTG.device_link.server import websocket_handler

        adapter = DeviceAdapter()
        with self.assertLogs("virtualrtg.adapter", level="DEBUG") as captured:
            async with serve(lambda ws: websocket_handler(ws, adapter, True), "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    coincident = state_update()
                    coincident["device"]["poses"]["tube"] = pose((0, 430, 0))
                    coincident["device"]["poses"]["detector"] = pose((0, -40, 0))
                    await ws.send(json.dumps(coincident))
                    accepted = json.loads(await ws.recv())
                    self.assertEqual(accepted["status"], "accepted")
                    self.assertEqual(adapter.state.revision, 1)

                    # A separate failure in optional verbose reporting must
                    # not change the accepted result or suppress its reply.
                    with patch("plugins.virtRTG.device_link.virtualrtg_geometry.map_device_geometry", side_effect=RuntimeError("diagnostic fault")):
                        await ws.send(json.dumps(message(
                            "device.pose", 3, object="tube", revision=2,
                            pose=pose((1, 430, 0)),
                        )))
                        accepted = json.loads(await ws.recv())
                    self.assertEqual(accepted["status"], "accepted")
                    self.assertEqual(adapter.state.revision, 2)
        output = "\n".join(captured.output)
        self.assertIn("geometry_consistent=False", output)
        self.assertIn("normal_ray_dot=undefined", output)
        self.assertIn("Geometry diagnostics failed after state acceptance: diagnostic fault", output)

    async def test_server_cancellation_closes_listener_and_active_client(self):
        import websockets
        from plugins.virtRTG.device_link.server import serve
        sock=socket.socket(); sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
        task=asyncio.create_task(serve("127.0.0.1",port))
        for _ in range(100):
            try:
                client=await websockets.connect(f"ws://127.0.0.1:{port}"); break
            except OSError: await asyncio.sleep(.01)
        else: self.fail("server did not start")
        task.cancel()
        await asyncio.wait_for(task,2)
        with self.assertRaises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(client.recv(),1)
        await client.close()
        replacement=socket.socket(); replacement.bind(("127.0.0.1",port)); replacement.close()

    async def test_unexpected_handler_errors_are_not_suppressed(self):
        from plugins.virtRTG.device_link.server import websocket_handler
        class BrokenAdapter:
            verbose=False
            def handle(self, message): raise RuntimeError("unexpected adapter defect")
        class FakeWebSocket:
            remote_address=("127.0.0.1",1234)
            async def __aiter__(self):
                yield json.dumps(message("device.hello",1))
            async def send(self, payload): self.sent=payload
        with self.assertRaisesRegex(RuntimeError,"unexpected adapter defect"):
            await websocket_handler(FakeWebSocket(),BrokenAdapter())


if __name__ == "__main__": unittest.main()
