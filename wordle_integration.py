import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.feature_extraction.text import CountVectorizer
import os

# Load the full Wordle dataset
with open("wordle_words.txt", "r") as f:
    word_list = [line.strip() for line in f.readlines() if len(line.strip()) == 5]

# Convert words into vector representations
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1))
X_words = vectorizer.fit_transform(word_list).toarray()
word_to_index = {word: i for i, word in enumerate(word_list)}


# Wordle feedback function
def get_feedback(guess, target):
    feedback = [0] * 5
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback[i] = 2  # Correct letter & position
        elif g in target:
            feedback[i] = 1  # Correct letter but wrong position
    return tuple(feedback)


# Get possible words based on previous guesses and feedback
def get_possible_words(guesses, feedbacks):
    possible_words = word_list.copy()
    for guess, feedback in zip(guesses, feedbacks):
        possible_words = [word for word in possible_words if get_feedback(guess, word) == feedback]
    return possible_words


# Define Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action


# Training parameters
learning_rate = 0.001
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.999  # Slower epsilon decay for better exploration
epsilon_min = 0.1
batch_size = 64
memory_size = 5000
num_episodes = 5000
save_interval = 1000  # Save model every 1000 episodes
model_path = "wordle_dqn_model.pth"

# Initialize DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(input_dim=5, output_dim=len(word_list)).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Experience Replay Memory
memory = deque(maxlen=memory_size)

# Check if a saved model exists
if os.path.exists(model_path):
    dqn.load_state_dict(torch.load(model_path))
    dqn.eval()  # Set to evaluation mode
    print("Pretrained model loaded successfully!")
else:
    print("No saved model found. Training a new model...")

    # Training loop
    for episode in range(num_episodes):
        target_word = random.choice(word_list)
        guesses = []
        feedbacks = []
        possible_words = word_list.copy()  # Track possible words
        attempts = 0
        state = np.zeros(5)  # State as a vector of zeros, for simplicity

        while attempts < 6:
            # Check if possible_words is empty, and reset if so
            if len(possible_words) == 0:
                possible_words = word_list.copy()  # Reset to full word list if filtered out
                print("Possible words list is empty, resetting to full word list.")

            # Choose action (epsilon-greedy)
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, len(possible_words) - 1)  # Explore
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action_index = torch.argmax(dqn(state_tensor)).item()  # Exploit

            # Ensure the action index is valid
            action_index = min(action_index, len(possible_words) - 1)

            guess = possible_words[action_index]
            feedback = get_feedback(guess, target_word)
            reward = feedback.count(2)  # Reward based on correct letters

            # Update possible words based on feedback
            possible_words = get_possible_words(guesses + [guess], feedbacks + [feedback])
            next_state = np.array([f for f in feedback])  # Use feedback as state transition

            # Store experience in memory
            memory.append((state, action_index, reward, next_state))

            # Train on minibatch
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

            # Move to next state
            state = next_state
            guesses.append(guess)
            feedbacks.append(feedback)
            attempts += 1

        # Decay epsilon more slowly
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Save the model at intervals
        if (episode + 1) % save_interval == 0:
            torch.save(dqn.state_dict(), model_path)
            print(f"Checkpoint saved at episode {episode + 1}.")

    # Final model save
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
test_word = random.choice(word_list)
print(f"Target Word: {test_word}")
print(play_wordle_internal(test_word))
