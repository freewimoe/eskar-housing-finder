#!/usr / bin / env python3
"""
Unit tests for User Feedback system
Tests feedback submission, validation, and storage mechanisms
"""

import unittest
import sqlite3
import os
import tempfile
from pathlib import Path

class TestUserFeedback(unittest.TestCase):
    """Test suite for User Feedback functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db_path = self.temp_file.name
        self.temp_file.close()
        self.sample_feedback = {
            'satisfaction': 5,
            'feedback_type': 'Property Search',
            'comments': 'Great ESK - specific features!'
        }

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_db_path):
            os.unlink(self.test_db_path)

    def test_feedback_database_creation(self):
        """Test feedback database table creation"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                satisfaction INTEGER,
                feedback_type TEXT,
                comments TEXT
            )
        """)

        # Verify table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)

        conn.close()

    def test_feedback_submission(self):
        """Test feedback submission process"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                satisfaction INTEGER,
                feedback_type TEXT,
                comments TEXT
            )
        """)

        # Insert test feedback
        cursor.execute("""
            INSERT INTO user_feedback (timestamp, satisfaction, feedback_type, comments)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            self.sample_feedback['satisfaction'],
            self.sample_feedback['feedback_type'],
            self.sample_feedback['comments']
        ))

        conn.commit()

        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM user_feedback")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)

        # Verify data integrity
        cursor.execute("SELECT satisfaction, feedback_type, comments FROM user_feedback WHERE id=1")
        result = cursor.fetchone()
        self.assertEqual(result[0], 5)
        self.assertEqual(result[1], 'Property Search')
        self.assertEqual(result[2], 'Great ESK - specific features!')

        conn.close()

    def test_feedback_validation(self):
        """Test feedback input validation"""
        # Test satisfaction score validation
        valid_scores = [1, 2, 3, 4, 5]
        invalid_scores = [0, 6, -1, 'five', None]

        for score in valid_scores:
            self.assertIn(score, range(1, 6))

        for score in invalid_scores:
            if isinstance(score, int):
                self.assertNotIn(score, range(1, 6))
            else:
                self.assertNotIsInstance(score, int)

    def test_feedback_types(self):
        """Test feedback type categories"""
        valid_types = [
            'Property Search',
            'Map Interface',
            'Price Predictions',
            'General Feedback',
            'Bug Report',
            'Feature Request'
        ]

        # Test all valid types are strings
        for feedback_type in valid_types:
            self.assertIsInstance(feedback_type, str)
            self.assertGreater(len(feedback_type), 0)

    def test_feedback_data_persistence(self):
        """Test feedback data persistence across sessions"""
        # Insert feedback in first session
        conn1 = sqlite3.connect(self.test_db_path)
        cursor1 = conn1.cursor()

        cursor1.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                satisfaction INTEGER,
                feedback_type TEXT,
                comments TEXT
            )
        """)

        cursor1.execute("""
            INSERT INTO user_feedback (timestamp, satisfaction, feedback_type, comments)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            4,
            'Map Interface',
            'Love the regional expansion'
        ))

        conn1.commit()
        conn1.close()

        # Verify persistence in second session
        conn2 = sqlite3.connect(self.test_db_path)
        cursor2 = conn2.cursor()

        cursor2.execute("SELECT COUNT(*) FROM user_feedback")
        count = cursor2.fetchone()[0]
        self.assertEqual(count, 1)

        cursor2.execute("SELECT satisfaction, comments FROM user_feedback WHERE id=1")
        result = cursor2.fetchone()
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1], 'Love the regional expansion')

        conn2.close()

    def test_feedback_aggregation(self):
        """Test feedback data aggregation for analytics"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                satisfaction INTEGER,
                feedback_type TEXT,
                comments TEXT
            )
        """)

        # Insert multiple feedback entries
        test_feedback = [
            (datetime.now().isoformat(), 5, 'Property Search', 'Excellent'),
            (datetime.now().isoformat(), 4, 'Property Search', 'Good'),
            (datetime.now().isoformat(), 5, 'Map Interface', 'Perfect'),
            (datetime.now().isoformat(), 3, 'Price Predictions', 'Okay')
        ]

        cursor.executemany("""
            INSERT INTO user_feedback (timestamp, satisfaction, feedback_type, comments)
            VALUES (?, ?, ?, ?)
        """, test_feedback)

        conn.commit()

        # Test average satisfaction
        cursor.execute("SELECT AVG(satisfaction) FROM user_feedback")
        avg_satisfaction = cursor.fetchone()[0]
        self.assertAlmostEqual(avg_satisfaction, 4.25, places=2)

        # Test feedback type distribution
        cursor.execute("SELECT feedback_type, COUNT(*) FROM user_feedback GROUP BY feedback_type")
        type_counts = dict(cursor.fetchall())
        self.assertEqual(type_counts['Property Search'], 2)
        self.assertEqual(type_counts['Map Interface'], 1)
        self.assertEqual(type_counts['Price Predictions'], 1)

        conn.close()

if __name__ == '__main__':
    print("💬 ESKAR Housing Finder - User Feedback Tests")
    print("=" * 50)

    # Run tests
    unittest.main(verbosity=2)
