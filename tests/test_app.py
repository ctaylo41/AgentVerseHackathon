import json
import unittest
from email.message import Message
from io import BytesIO
from types import SimpleNamespace

from robopod.app import AppState, Studio, make_handler


class AppTests(unittest.TestCase):
    def test_workflow_guards_and_single_operation(self):
        state=AppState()
        with self.assertRaises(ValueError): state.submit('execute')
        with self.assertRaises(ValueError): state.submit('plan','brief')
        state.submit('explore')
        self.assertTrue(state.snapshot()['busy'])
        with self.assertRaises(ValueError): state.submit('explore')
        self.assertEqual(state.commands.get_nowait(),('explore',''))
        state.submit('restart')
        self.assertTrue(state.cancel.is_set())
        self.assertEqual(state.snapshot()['phase'],'restarting')
        with self.assertRaises(ValueError): state.submit('restart')

    def test_snapshot_is_detached(self):
        state=AppState()
        state.log('A message')
        snap=state.snapshot()
        snap['events'].append('injected')
        self.assertEqual(state.snapshot()['events'],['A message'])
        for i in range(100):state.log(str(i))
        self.assertEqual(len(state.snapshot()['events']),80)

    def test_stale_planner_result_ignored_after_restart(self):
        state=AppState()
        studio=Studio(state,'unused')
        studio.epoch=2
        studio.plan_results.put((1,None,{'shots':[]},None))
        studio.poll_plans()
        self.assertIsNone(state.snapshot()['plan'])

    def test_local_http_action_auth_and_origin(self):
        state=AppState()
        handler=object.__new__(make_handler(state))
        handler.server=SimpleNamespace(server_port=8765)
        handler.path='/api/action'
        payload=json.dumps({'action':'explore'}).encode()
        result=[]
        handler.send=lambda code,body,kind='application/json':result.append((code,body))
        handler.headers=Message()
        handler.headers['Host']='127.0.0.1:8765'
        handler.headers['Content-Length']=str(len(payload))
        handler.rfile=BytesIO(payload)
        handler.do_POST()
        self.assertEqual(result[-1][0],403)
        handler.headers['X-Robopod-Token']=state.token
        handler.headers['Origin']='https://external.example'
        handler.do_POST()
        self.assertEqual(result[-1][0],403)
        handler.headers.replace_header('Origin','http://127.0.0.1:8765')
        handler.do_POST()
        self.assertEqual(result[-1][0],202)
        handler.rfile=BytesIO(payload)
        handler.do_POST()
        self.assertEqual(result[-1][0],409)
