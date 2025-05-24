import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Tic Tac Toe Spielumgebung
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board[:] = 0
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action, player):
        if self.done:
            raise ValueError("Spiel ist vorbei")

        i, j = action
        if self.board[i, j] != 0:
            return self.get_state(), -10, True

        self.board[i, j] = player
        if self.check_winner(player):
            self.done = True
            self.winner = player
            return self.get_state(), 1, True
        elif not self.available_actions():
            self.done = True
            return self.get_state(), 0.5, True

        return self.get_state(), 0, False

    def check_winner(self, player):
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def render_gui(self):
        fig, ax = plt.subplots()
        ax.set_xticks([0.5, 1.5], minor=False)
        ax.set_yticks([0.5, 1.5], minor=False)
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.grid(True, which='major', color='black', linewidth=2)

        for i in range(3):
            for j in range(3):
                value = self.board[i, j]
                if value == 1:
                    ax.plot(j, 2 - i, 'x', markersize=20, markeredgewidth=2, color='blue')
                elif value == -1:
                    circle = plt.Circle((j, 2 - i), 0.3, fill=False, color='red', linewidth=2)
                    ax.add_patch(circle)

        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()

# Q-Learning-Agent
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table[(state, action)]

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            q_values = [self.get_q(state, a) for a in actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_actions):
        max_q_next = max([self.get_q(next_state, a) for a in next_actions], default=0)
        old_value = self.q_table[(state, action)]
        new_value = old_value + self.alpha * (reward + self.gamma * max_q_next - old_value)
        self.q_table[(state, action)] = new_value

# Training mit Lernkurve
def train(agent, episodes=10000):
    env = TicTacToe()
    results = {"win": 0, "draw": 0, "loss": 0}

    win_rates = []
    draw_rates = []
    loss_rates = []

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            actions = env.available_actions()
            action = agent.choose_action(state, actions)

            next_state, reward, done = env.step(action, player=1)

            if done:
                agent.update(state, action, reward, next_state, [])
                if reward == 1:
                    results["win"] += 1
                elif reward == 0.5:
                    results["draw"] += 1
                else:
                    results["loss"] += 1
                break

            opp_actions = env.available_actions()
            opp_action = random.choice(opp_actions)
            next_state2, reward2, done = env.step(opp_action, player=-1)

            if done:
                agent.update(state, action, -1 * reward2, next_state2, [])
                if reward2 == 1:
                    results["loss"] += 1
                elif reward2 == 0.5:
                    results["draw"] += 1
                else:
                    results["win"] += 1
                break

            next_actions = env.available_actions()
            agent.update(state, action, reward, next_state2, next_actions)
            state = next_state2

        if (episode + 1) % 1000 == 0:
            total = sum(results.values())
            win_rates.append(results["win"] / total)
            draw_rates.append(results["draw"] / total)
            loss_rates.append(results["loss"] / total)
            print(f"Episode {episode+1}: Wins {results['win']}, Draws {results['draw']}, Losses {results['loss']}")
            results = {"win": 0, "draw": 0, "loss": 0}

    x = [i * 1000 for i in range(1, len(win_rates) + 1)]
    plt.plot(x, win_rates, label="Win Rate")
    plt.plot(x, draw_rates, label="Draw Rate")
    plt.plot(x, loss_rates, label="Loss Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Rate")
    plt.title("Lernkurve des Q-Learning-Agenten")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    agent = QLearningAgent()
    train(agent, episodes=10000)

    # Visualisierung eines Beispielbretts
    env = TicTacToe()
    env.board[0, 0] = 1
    env.board[1, 1] = -1
    env.render_gui()
