import unittest
import weather_lib as wl



url = 'http://api.weatherbit.io/v2.0/history/hourly'


class TestCNN(unittest.TestCase):
    def test_request(self):
        self.assertEqual(wl.make_request(wl.dataTest(), url)[0], 200)
        self.assertEqual(len(wl.make_request(wl.dataTest(), url)[1]), 24)
    
    def test_save_frames(self):
        self.assertEqual(len(wl.save_frames('frames_test')), 3)
        


if __name__ == '__main__':
    unittest.main()