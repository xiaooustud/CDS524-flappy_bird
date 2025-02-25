import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import os
from datetime import datetime
import math
import matplotlib.pyplot as plt


class FlappyBird:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Flappy Bird Q-Learning')

        # Game parameters
        self.bird_x = 150
        self.bird_size = 30
        self.pipe_width = 60
        self.pipe_gap = 230
        self.pipe_speed = 2.0
        self.gravity = 0.3
        self.jump_strength = -6
        self.terminal_velocity = 8
        self.max_pipes = 15  # Maximum number of pipes allowed

        # Fixed pipe configuration
        self.pipe_template = self.generate_pipe_template()

        # Colors
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.SKY_BLUE = (135, 206, 235)

        # Performance metrics
        self.max_score = 0
        self.total_frames = 0

        self.reset_game()

    def generate_pipe_template(self):
        """
        Generates a fixed pattern of pipes with regular intervals.

        Returns:
            list: A list of dictionaries containing pipe configurations.
                  Each dictionary contains:
                  - 'x': x-coordinate of the pipe
                  - 'gap_y': y-coordinate of the gap center
        """
        pipes = []
        base_heights = [250, 300, 200, 350, 275] * 3  # Extend pattern to 15 pipes
        x_start = 400
        x_interval = 300

        for i in range(self.max_pipes):
            pipes.append({
                'x': x_start + i * x_interval,
                'gap_y': base_heights[i]
            })
        return pipes

    def reset_game(self):
        """
        Resets the game state to initial conditions.

        Returns: tuple: Initial state representation
        """
        self.bird_y = self.height // 2
        self.bird_velocity = 0
        self.pipes = [pipe.copy() for pipe in self.pipe_template]
        self.score = 0
        self.alive = True
        self.frames = 0  # Number of frames in current episode

        return self.get_state()

    def get_state(self):
        """
        Creates an improved state representation for the agent.

        Returns: tuple: (horizontal_distance, vertical_distance, velocity, next_pipe_distance)
        """
        if not self.pipes:
            return (0, 0, 0, 0)

        # Find nearest pipe and next pipe
        nearest_pipe = None
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                if nearest_pipe is None:
                    nearest_pipe = pipe
                elif next_pipe is None and pipe['x'] > nearest_pipe['x']:
                    next_pipe = pipe
                    break

        if nearest_pipe is None:
            return (0, 0, 0, 0)

        # Calculate precise state values
        horizontal_distance = (nearest_pipe['x'] - self.bird_x) / self.width
        vertical_distance = (nearest_pipe['gap_y'] - self.bird_y) / self.height
        velocity = self.bird_velocity / self.terminal_velocity

        # Add next pipe information
        next_pipe_distance = 1.0
        if next_pipe:
            next_pipe_distance = (next_pipe['x'] - self.bird_x) / self.width

        # Discretize state space while maintaining detail
        h_dist = int(horizontal_distance * 15)  # More horizontal distance intervals
        v_dist = int(vertical_distance * 15)  # More vertical distance intervals
        vel = int(velocity * 10)  # More velocity intervals
        next_dist = int(next_pipe_distance * 10)

        return (h_dist, v_dist, vel, next_dist)

    def step(self, action):
        """
        Executes one step in the environment based on the given action.

        Args: action: The action to take (0: do nothing, 1: jump)

        Returns: tuple: (new_state, reward, done)
        """
        self.frames += 1
        reward = 0

        # Base survival reward
        reward += 0.1

        if action == 1:
            self.bird_velocity = self.jump_strength
            reward -= 0.1  # Small penalty for jumping

        # Update bird position and velocity
        self.bird_velocity = min(self.bird_velocity + self.gravity, self.terminal_velocity)
        self.bird_y += self.bird_velocity

        # Move pipes
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed

        # Check pipe passing
        for pipe in self.pipes[:]:
            if pipe['x'] + self.pipe_width < self.bird_x and 'passed' not in pipe:
                pipe['passed'] = True
                self.score += 1
                reward += 25.0

                # Check if completed all pipes
                if self.score >= self.max_pipes:
                    reward += 80.0  # Bonus reward
                    self.alive = False
                    return self.get_state(), reward, True

        # Reset pipes
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
            new_pipe = self.pipe_template[0].copy()
            new_pipe['x'] = self.pipes[-1]['x'] + 300
            self.pipes.append(new_pipe)

        # Collision detection
        if self.check_collision():
            # Adjust penalty based on survival time
            survival_factor = min(self.frames / 1000, 1.0)
            reward = -15 * survival_factor  # Dynamic penalty
            self.alive = False
            return self.get_state(), reward, True

        # Reward for staying near pipe center
        nearest_pipe = min((p for p in self.pipes if p['x'] + self.pipe_width > self.bird_x),
                           key=lambda p: p['x'])
        vertical_distance = abs(nearest_pipe['gap_y'] - self.bird_y)
        if vertical_distance < self.pipe_gap / 4:
            reward += 0.5

        return self.get_state(), reward, False

    def check_collision(self):
        """
        Implements improved collision detection between bird and obstacles.

        Returns: bool: True if collision detected, False otherwise
        """
        # Check boundary collisions
        if self.bird_y - self.bird_size / 2 < 0 or \
                self.bird_y + self.bird_size / 2 > self.height:
            return True

        # Precise collision detection using rectangles
        bird_rect = pygame.Rect(
            self.bird_x - self.bird_size / 2,
            self.bird_y - self.bird_size / 2,
            self.bird_size,
            self.bird_size
        )

        for pipe in self.pipes:
            upper_pipe = pygame.Rect(
                pipe['x'],
                0,
                self.pipe_width,
                pipe['gap_y'] - self.pipe_gap / 2
            )
            lower_pipe = pygame.Rect(
                pipe['x'],
                pipe['gap_y'] + self.pipe_gap / 2,
                self.pipe_width,
                self.height
            )

            if bird_rect.colliderect(upper_pipe) or \
                    bird_rect.colliderect(lower_pipe):
                return True

        return False

    def render(self):
        """
        Renders the game state with improved visuals.
        """
        self.screen.fill(self.SKY_BLUE)

        # Draw pipes
        for pipe in self.pipes:
            # Upper pipe
            pygame.draw.rect(self.screen, self.GREEN,
                             (pipe['x'],
                              0,
                              self.pipe_width,
                              pipe['gap_y'] - self.pipe_gap / 2))
            # Lower pipe
            pygame.draw.rect(self.screen, self.GREEN,
                             (pipe['x'],
                              pipe['gap_y'] + self.pipe_gap / 2,
                              self.pipe_width,
                              self.height))

        # Draw bird with rotation effect
        bird_surface = pygame.Surface((self.bird_size, self.bird_size),
                                      pygame.SRCALPHA)
        pygame.draw.circle(bird_surface, self.YELLOW,
                           (self.bird_size // 2, self.bird_size // 2),
                           self.bird_size // 2)

        # Calculate rotation based on velocity
        rotation = math.degrees(math.atan(self.bird_velocity / 10))
        rotated_bird = pygame.transform.rotate(bird_surface, -rotation)
        self.screen.blit(rotated_bird,
                         (self.bird_x - rotated_bird.get_width() // 2,
                          self.bird_y - rotated_bird.get_height() // 2))

        # Display game information
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, self.WHITE)
        max_score_text = font.render(f'Max Score: {self.max_score}',
                                     True, self.WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()


class DQN(nn.Module):
    """
    Deep Q-Network architecture
    """

    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    """
    DQN Agent implementation with experience replay and target network
    """

    def __init__(self, state_size, n_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_size = state_size
        self.n_actions = n_actions

        self.policy_net = DQN(state_size, n_actions).to(device)
        self.target_net = DQN(state_size, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=50000)

        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 1000
        self.steps_done = 0

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(0)[1].item()
        else:
            return random.randrange(self.n_actions)

    def train_step(self):
        """Perform one step of training"""
        if len(self.memory) < self.batch_size:
            return None

        transitions = random.sample(self.memory, self.batch_size)

        batch_state = torch.FloatTensor(np.array([t[0] for t in transitions])).to(self.device)
        batch_action = torch.LongTensor(np.array([t[1] for t in transitions])).to(self.device)
        batch_reward = torch.FloatTensor(np.array([t[2] for t in transitions])).to(self.device)
        batch_next_state = torch.FloatTensor(np.array([t[3] for t in transitions])).to(self.device)
        batch_done = torch.FloatTensor(np.array([t[4] for t in transitions])).to(self.device)

        current_q_values = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1))
        next_q_values = self.target_net(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + (1 - batch_done) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()


def get_state(game):
    """
    Convert game state to neural network input format

    Args: game: FlappyBird game instance

    Returns: numpy.array: Normalized state values
    """
    # Get game state
    bird_y = game.bird_y / game.height  # Normalize bird height
    bird_velocity = game.bird_velocity / game.terminal_velocity  # Normalize velocity

    # Get nearest pipe information
    nearest_pipe = None
    for pipe in game.pipes:
        if pipe['x'] + game.pipe_width > game.bird_x:
            nearest_pipe = pipe
            break

    if nearest_pipe is None:
        pipe_dist_x = 1.0
        pipe_top_y = 0.5
        pipe_bottom_y = 0.5
    else:
        pipe_dist_x = (nearest_pipe['x'] - game.bird_x) / game.width
        pipe_top_y = (nearest_pipe['gap_y'] - game.pipe_gap / 2) / game.height
        pipe_bottom_y = (nearest_pipe['gap_y'] + game.pipe_gap / 2) / game.height

    state = np.array([bird_y, bird_velocity, pipe_dist_x, pipe_top_y, pipe_bottom_y])
    return state


class DQNTrainer:
    """
    Handles the training process of the DQN agent
    """
    def __init__(self, game):
        self.game = game
        self.agent = DQNAgent(state_size=5, n_actions=2)

        # Create directories for saving models and plots
        for directory in ['models', 'plots']:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Initialize training data collection
        self.training_data = {
            'episodes': [],
            'scores': [],
            'rewards': [],
            'losses': [],
            'epsilons': [],
            'moving_avg_scores': []
        }

    def save_training_plots(self, episode):
        """Save training progress plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_data['episodes'], self.training_data['losses'], 'b-', label='Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/loss_curve_{timestamp}.png')
        plt.close()

        # Create learning curve (scores)
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_data['episodes'], self.training_data['scores'], 'g-',
                 label='Score', alpha=0.5)
        plt.plot(self.training_data['episodes'], self.training_data['moving_avg_scores'], 'r-',
                 label='Moving Average Score')
        plt.title('Learning Curve - Score Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/learning_curve_{timestamp}.png')
        plt.close()

        # Create convergence curve (epsilon and rewards)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Epsilon', color='tab:blue')
        ax1.plot(self.training_data['episodes'], self.training_data['epsilons'], 'b-',
                 label='Epsilon')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Total Reward', color='tab:orange')
        ax2.plot(self.training_data['episodes'], self.training_data['rewards'], 'orange',
                 label='Total Reward')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        plt.title('Convergence Analysis')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.savefig(f'plots/convergence_curve_{timestamp}.png')
        plt.close()

    def calculate_moving_average(self, values, window=10):
        """Calculate moving average of values"""
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')

    def train(self, episodes=200):
        """
        Main training loop
        Args: episodes: Number of episodes to train
        """
        best_score = 0
        episode_losses = []

        for episode in range(episodes):
            self.game.reset_game()
            state = get_state(self.game)
            total_reward = 0
            episode_loss = []

            while self.game.alive:
                # Select and perform action
                action = self.agent.select_action(state)
                _, reward, done = self.game.step(action)
                next_state = get_state(self.game)

                # Store transition and train
                self.agent.store_transition(state, action, reward, next_state, done)
                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                state = next_state
                total_reward += reward

                # Render game
                self.game.render()
                pygame.time.delay(20)

                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            # Record training data
            self.training_data['episodes'].append(episode + 1)
            self.training_data['scores'].append(self.game.score)
            self.training_data['rewards'].append(total_reward)
            self.training_data['epsilons'].append(self.agent.epsilon)

            # Calculate and record average loss
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            self.training_data['losses'].append(avg_loss)

            # Calculate moving average score
            if len(self.training_data['scores']) >= 10:
                moving_avg = self.calculate_moving_average(
                    self.training_data['scores'])[-1]
            else:
                moving_avg = np.mean(self.training_data['scores'])
            self.training_data['moving_avg_scores'].append(moving_avg)

            # Update best score and save model
            if self.game.score > best_score:
                best_score = self.game.score
                self.save_model('best_model')

            # Save checkpoint and plots every 50 episodes
            if (episode + 1) % 50 == 0:
                self.save_model(f'checkpoint_episode_{episode + 1}')
                self.save_training_plots(episode + 1)

            print(f"Episode: {episode + 1}, Score: {self.game.score}, "
                  f"Total Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}, "
                  f"Loss: {avg_loss:.4f}")

        # Save final plots after training
        self.save_training_plots(episodes)

    def save_model(self, name):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'models/{name}_{timestamp}.pth'
        torch.save({
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
        }, filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.epsilon = checkpoint['epsilon']


def main():
    """Main entry point of the program"""
    game = FlappyBird()
    trainer = DQNTrainer(game)
    trainer.train()


if __name__ == "__main__":
    main()