import gzip,json,tempfile,unittest
from pathlib import Path
from recover_capture import recover
from replay_combined import run

class RecoveryTests(unittest.TestCase):
    def test_complete_messages_preserved_without_inventing_final_metadata(self):
        rows=[dict(kind='metadata',t=10.,wall=100.,duration=1800,metadata={}),
              dict(kind='timer',t=12.,wall=102.)]
        raw=''.join(json.dumps(r)+'\n' for r in rows).encode()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'original.gz';src.write_bytes(gzip.compress(raw)[:-8])
            before=src.read_bytes();out=root/'recovered'
            summary=recover(src,out)
            self.assertEqual(src.read_bytes(),before)
            self.assertEqual(gzip.decompress((out/'feed.jsonl.gz').read_bytes()),raw)
            self.assertIsNone(summary['end_metadata'])
            self.assertEqual(summary['planned_duration_seconds'],1800)
            self.assertEqual(summary['recovered_seconds'],2)
            with self.assertRaises(RuntimeError):run(out)
            result=run(out,allow_incomplete=True)
            self.assertEqual(result['planned_duration_seconds'],1800)
            self.assertTrue(result['incomplete_capture'])
            self.assertEqual(result['fee_validation'],'start_only_end_unavailable')

    def test_partial_last_json_is_not_a_market_message(self):
        row=dict(kind='metadata',t=10.,wall=100.,duration=1800,metadata={})
        raw=(json.dumps(row)+'\n'+ '{"kind":"message"').encode()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'original.gz';src.write_bytes(gzip.compress(raw)[:-8])
            summary=recover(src,root/'recovered')
            self.assertEqual(summary['recovery']['complete_records'],1)
            self.assertGreater(summary['recovery']['trailing_incomplete_bytes_discarded'],0)

if __name__=='__main__':unittest.main()
