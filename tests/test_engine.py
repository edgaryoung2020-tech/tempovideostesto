import json
import subprocess as sp
import tempfile
import unittest
from pathlib import Path
from engine import validate_plan, render, probe

class PlanTests(unittest.TestCase):
    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan([dict(start=0,end=2,text='A'),dict(start=1,end=3,text='B')],5)
    def test_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError): validate_plan([dict(start=1,end=7,text='A')],5)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError): validate_plan([dict(start=float('nan'),end=2,text='A')],5)
    def test_cyrillic_render_preserves_audio(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'in.mp4'; out=Path(td)/'out.mp4'
            sp.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=0x343c50:s=360x640:r=30:d=3','-f','lavfi','-i','sine=frequency=440:duration=3','-c:v','libx264','-c:a','aac',str(src)],check=True)
            render(src,out,[dict(start=.4,end=2.5,kind='stat',text='30 секунд',label='Исходный звук')])
            def packets(p):
                data=json.loads(sp.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_packets','-show_data_hash','sha256','-of','json',str(p)]))
                return [v['data_hash'] for v in data['packets']]
            self.assertEqual(packets(src),packets(out))
            streams=probe(out)['streams']; self.assertEqual(streams[0]['width'],360)
            def frame(p):
                return sp.check_output(['ffmpeg','-v','error','-ss','1','-i',str(p),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
            self.assertNotEqual(frame(src),frame(out))

if __name__=='__main__': unittest.main()
