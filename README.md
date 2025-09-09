# Hand Cricket: AI-Powered Game 🏏👋

Welcome to **Hand Cricket**, an interactive game that brings the classic game of hand cricket to life using computer vision and AI. Play against a friend by using hand gestures, which are detected in real time to determine the outcome of each ball.

-----

## 🧐 What is This?

This project is a modern twist on the popular game of hand cricket. Instead of manually showing fingers, you use a webcam to display your hand gestures. An AI model, powered by **MediaPipe** and a **Random Forest classifier**, recognizes your gestures to determine the runs scored or if a player is out.

The game is designed for two players, each with their own camera (or camera index). The first player to get out ends their inning, and the second player then bats to chase the score.

-----

## 🛠️ Requirements

Before you can play, make sure you have the necessary libraries installed.

  * **Python 3.x**
  * **OpenCV (`cv2`)**: For accessing the webcam.
  * **MediaPipe (`mediapipe`)**: For real-time hand landmark detection.
  * **scikit-learn (`sklearn`)**: The Random Forest model is from here.
  * **NumPy (`numpy`)**: For numerical operations.
  * **Pickle (`pickle`)**: To load the pre-trained model.

You can install these dependencies using pip:

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

-----

## 🚀 How to Run

1.  **Clone the Repository**: Get the code onto your local machine.

2.  **Load the Model**: Ensure you have the `hand_model.pkl` file in the same directory as the main script. This file contains the pre-trained Random Forest model that recognizes the hand gestures.

3.  **Run the Script**: Open your terminal or command prompt, navigate to the project directory, and run the script.

    ```bash
    python your_script_name.py
    ```

4.  **Game Setup**: Follow the on-screen prompts to set up the game:

      * Enter the names of the two players.
      * One player will call "Head" or "Tail" for the toss.
      * The toss winner chooses to "bat" or "bowl" first.

5.  **Start Playing**:

      * Two camera windows will open, one for the **batter** and one for the **bowler**.
      * To play a shot or bowl a ball, make a clear hand gesture in front of your respective camera.
      * The gestures recognized are **0, 1, 2, 3, 4,** and **6**.
      * When the batter's and bowler's gestures match, the batter is **out**\! 🏏
      * To get ready for the next ball, simply make a **closed fist** gesture. The game will recognize this as "Ready for next gesture".

-----

## 🖼️ Game in Action

The game uses real-time hand tracking. You will see a skeleton overlay on your hand, showing how MediaPipe detects your landmarks.

-----

## 💻 Code Breakdown

  * `camera()`: A function that runs in a separate process for each player's camera. It captures video, detects hand landmarks using MediaPipe, and uses the loaded `hand_model.pkl` to predict the gesture.
  * `play_inning()`: The main game loop function. It uses **`multiprocessing`** to handle both cameras simultaneously, ensuring a smooth, real-time experience without lag. It compares the batter's and bowler's gestures to calculate runs or determine an "out."
  * `main` block (`if __name__ == '__main__':`): Handles the game flow, including the toss, innings, and final score calculation.

-----

## 🤝 Contribution

Feel free to fork this repository, suggest improvements, or submit pull requests. This project can be extended with features like:

  * More gestures (e.g., 5).
  * Sound effects for runs and wickets.
  * A user interface (GUI) for a more polished experience.
  * Training the model on a wider range of hand gestures for improved accuracy.
