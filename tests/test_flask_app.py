import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        """Ensure the home page loads and title is correct"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Calories Burnt Predictor</title>', response.data)

    def test_predict_page(self):
        """Test prediction with sample form input"""
        response = self.client.post('/predict', data=dict(
            Gender="male",
            Age=25,
            Height=170,
            Weight=70,
            Duration=30,
            Heart_Rate=120,
            Body_Temp=98.6
        ))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'You burned', response.data)

if __name__ == '__main__':
    unittest.main()
