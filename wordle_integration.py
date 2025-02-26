import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.feature_extraction.text import CountVectorizer
import os

# Load the full Wordle dataset from file
with open("wordle_words.txt", "r") as f:
    word_list = [line.strip() for line in f.readlines() if len(line.strip()) == 5]  # Ensure only 5-letter words

# Convert words into vector representations using character counts
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1))
X_words = vectorizer.fit_transform(word_list).toarray()  # Convert words into numerical format
word_to_index = {word: i for i, word in enumerate(word_list)}  # Map words to indices


# Function to compute Wordle-style feedback for a given guess
def get_feedback(guess, target):
    """
    Generates feedback for the guessed word compared to the target word.
    - 2 (🟩): Letter is correct and in the correct position.
    - 1 (🟨): Letter is correct but in the wrong position.
    - 0 (⬛): Letter is not in the target word.
    """
    feedback = [0] * 5
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback[i] = 2  # Correct letter in correct position
        elif g in target:
            feedback[i] = 1  # Correct letter in wrong position
    return tuple(feedback)


# Function to filter possible words based on previous guesses and feedback
def get_possible_words(guesses, feedbacks):
    """
    Narrows down possible words based on previous guesses and received feedback.
    Only words that match the provided feedback survive.
    """
    possible_words = word_list.copy()
    for guess, feedback in zip(guesses, feedbacks):
        possible_words = [word for word in possible_words if get_feedback(guess, word) == feedback]
    return possible_words


# Define Deep Q-Network (DQN) architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Defines a Deep Q-Network with three fully connected layers.
        - input_dim: Number of input features (5 in this case, corresponding to feedback states).
        - output_dim: Number of possible actions (i.e., number of words in word_list).
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        Forward pass through the neural network.
        Uses ReLU activation for the hidden layers.
        Outputs Q-values for all possible words.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action


# Training parameters
learning_rate = 0.001
gamma = 0.9  # Discount factor for future rewards
epsilon = 1.0  # Exploration rate (starts high and decays)
epsilon_decay = 0.999  # Decay rate for epsilon to encourage exploitation
epsilon_min = 0.1  # Minimum epsilon value
batch_size = 64  # Number of experiences per training step
memory_size = 5000  # Size of experience replay memory
num_episodes = 5000  # Total number of training episodes
save_interval = 1000  # Save model every 1000 episodes
model_path = "wordle_dqn_model.pth"  # Path to save/load the model

# Initialize DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
dqn = DQN(input_dim=5, output_dim=len(word_list)).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Experience Replay Memory
memory = deque(maxlen=memory_size)

# Load pre-trained model if it exists
if os.path.exists(model_path):
    dqn.load_state_dict(torch.load(model_path))
    dqn.eval()  # Set to evaluation mode
    print("Pretrained model loaded successfully!")
else:
    print("No saved model found. Training a new model...")

    # Training loop
    for episode in range(num_episodes):
        target_word = random.choice(word_list)  # Select a random target word
        guesses = []
        feedbacks = []
        possible_words = word_list.copy()
        attempts = 0
        state = np.zeros(5)  # Initialize state representation

        while attempts < 6:  # Wordle allows up to 6 attempts
            # Ensure there's a valid word list to choose from
            if len(possible_words) == 0:
                possible_words = word_list.copy()  # Reset if empty
                print("Possible words list is empty, resetting.")

            # Choose action using epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, len(possible_words) - 1)  # Random choice (explore)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action_index = torch.argmax(dqn(state_tensor)).item()  # Choose best action (exploit)

            # Ensure action index is valid
            action_index = min(action_index, len(possible_words) - 1)

            guess = possible_words[action_index]
            feedback = get_feedback(guess, target_word)
            reward = feedback.count(2)  # Reward = count of correct letters

            # Update possible words based on feedback
            possible_words = get_possible_words(guesses + [guess], feedbacks + [feedback])
            next_state = np.array([f for f in feedback])  # Convert feedback to state

            # Store experience in replay memory
            memory.append((state, action_index, reward, next_state))

            # Train on minibatch if enough data is available
            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states = zip(*minibatch)

                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)

                current_Q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_Q = dqn(next_states).max(1)[0].detach()
                target_Q = rewards + (gamma * next_Q)

                loss = loss_fn(current_Q, target_Q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            guesses.append(guess)
            feedbacks.append(feedback)
            attempts += 1

        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Save model periodically
        if (episode + 1) % save_interval == 0:
            torch.save(dqn.state_dict(), model_path)
            print(f"Checkpoint saved at episode {episode + 1}.")

    # Save final trained model
    torch.save(dqn.state_dict(), model_path)
    print("Training completed! Model saved.")


def play_wordle_internal(target_word, game_number=1345):
    state = np.zeros(5)  # Initial state (no feedback yet)
    guesses = []
    feedbacks = []

    # Start with the formatted header
    result_header = f"Wordle {game_number} "
    result_body = ""

    # Keep track of possible words based on feedback
    possible_words = word_list.copy()

    # Make up to 6 guesses
    for attempt in range(6):
        # Select action based on epsilon-greedy strategy
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_index = torch.argmax(dqn(state_tensor)).item()

        # Ensure the action is within the valid range
        action_index = min(action_index, len(possible_words) - 1)

        guess = possible_words[action_index]
        print(f"Guess: {guess}")

        feedback = get_feedback(guess, target_word)

        # Convert feedback to colored emojis
        emoji_feedback = "".join(["🟩" if f == 2 else "🟨" if f == 1 else "⬛" for f in feedback])

        # Append guess with emoji feedback to the result string
        result_body += f"\n{emoji_feedback}"

        # Check if the guess is correct
        if feedback == (2, 2, 2, 2, 2):  # All letters are correct
            result_header = result_header + f" {attempt + 1}/6"
            result = result_header + f"\r\n{result_body.strip()} "  # Add number of attempts at the end
            return result

        # Update the possible words list based on feedback
        possible_words = get_possible_words(guesses + [guess], feedbacks + [feedback])

        # Update state with the feedback of the current guess
        state = np.array([f for f in feedback])

        # Keep track of guesses and feedbacks for future decisions
        guesses.append(guess)
        feedbacks.append(feedback)

    # If the AI fails to guess in 6 attempts
    result_header = result_header + " X/6"
    result = result_header + f"\r\n{result_body.strip()}"  # Indicate failure with X/6
    return result


# Test the trained model
# test_word = random.choice(word_list)
# print(f"Target Word: {test_word}")
# print(play_wordle_internal(test_word))
