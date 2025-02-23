import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.feature_extraction.text import CountVectorizer

# Load a small Wordle word list (for simplicity)
word_list = ["apple", "grape", "peach", "mango", "melon", "berry", "chess", "table", "chair", "couch"]

# Convert words into feature vectors
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1))
X_words = vectorizer.fit_transform(word_list).toarray()
word_to_index = {word: i for i, word in enumerate(word_list)}

# Wordle feedback simulation function
def get_feedback(guess, target):
    feedback = [0] * 5
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback[i] = 2  # Correct letter & position
        elif g in target:
            feedback[i] = 1  # Correct letter but wrong position
    return tuple(feedback)

# Neural Network for Deep Q-Learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

# Training parameters
learning_rate = 0.001
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 32
memory_size = 1000
num_episodes = 1000

# Initialize DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(input_dim=5, output_dim=len(word_list)).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Experience Replay Memory
memory = deque(maxlen=memory_size)

# Training loop
for episode in range(num_episodes):
    target_word = random.choice(word_list)
    state = (0, 0, 0, 0, 0)  # Initial state (no feedback yet)
    attempts = 0

    while attempts < 6:
        # Choose action (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, len(word_list) - 1)  # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action_index = torch.argmax(dqn(state_tensor)).item()  # Exploit

        guess = word_list[action_index]
        feedback = get_feedback(guess, target_word)
        reward = sum(feedback)  # More correct letters = higher reward

        # Store experience in memory
        memory.append((state, action_index, reward, feedback))

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
        state = feedback
        attempts += 1

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training completed!")

# Function to play a Wordle game with the trained model
def play_wordle(target_word):
    state = (0, 0, 0, 0, 0)
    for attempt in range(6):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_index = torch.argmax(dqn(state_tensor)).item()
        guess = word_list[action_index]
        feedback = get_feedback(guess, target_word)
        print(f"Attempt {attempt+1}: {guess} -> Feedback: {feedback}")

        if feedback == (2, 2, 2, 2, 2):
            print("Word guessed correctly!")
            return
        state = feedback

    print("Failed to guess the word.")

# Test the trained model
test_word = random.choice(word_list)
print(f"Target Word: {test_word}")
play_wordle(test_word)
