library(ReinforcementLearning)
library(dplyr)
library(ggplot2)

# Load dataset from local file (update the path accordingly)
adjusted_data <- read.csv("C:/Users/scricco/Desktop/expanded_synthetic_financial_dataset_consistent.csv")

# Define country risk score (proxy based on market volatility and unemployment)
adjusted_data <- adjusted_data %>%
  mutate(Country_Risk_Score = 0.5 * Market_Volatility + 0.5 * Unemployment)

# Initialize P2R and SREP columns (assuming baseline values)
adjusted_data <- adjusted_data %>%
  mutate(
    P2R = runif(n(), 1.5, 3.0),  # Pillar 2 Requirement between 1.5% and 3%
    SREP = runif(n(), 3.0, 5.0)   # SREP leverage requirement between 3% and 5%
  )

data <- adjusted_data %>%
  mutate(
    Country_Risk_Category = ifelse(Country_Risk_Score > median(Country_Risk_Score), "High", "Low"),
    State = paste0(
      "Risk_", Country_Risk_Category, "_",
      "GDP_", ifelse(GDP_Growth > 2, "High", "Low"), "_",
      "NPL_", ifelse(NPL_Ratio < 5, "Low", "High"), "_",
      "Credit_", ifelse(Credit_Growth > 5, "High", "Low"), "_",
      "CCyB_", as.character(CCyB)
    ),
    Action = sample(c("Increase_P2R", "Decrease_P2R", "Increase_SREP", "Decrease_SREP"), n(), replace = TRUE),
    Reward = 15 - abs(Credit_Growth - 5) - (NPL_Ratio / 4) - 
      abs(lag(Action, default = "Decrease_P2R") != Action) - 
      (0.2 * abs(P2R - lag(P2R, default = first(P2R)))) - 
      (0.2 * abs(SREP - lag(SREP, default = first(SREP))))  # Penalizing large capital requirement adjustments and favoring stability
  )

# Create next state representation
data <- data %>% mutate(NextState = lead(State)) %>% na.omit()

# Define adaptive epsilon decay function
adaptive_epsilon <- function(iteration, rewards, initial_epsilon = 0.7, min_epsilon = 0.2, decay_rate = 0.99) {
  return(max(min_epsilon, initial_epsilon * (decay_rate ^ iteration)))
}

# Initialize the Q-learning model
model <- ReinforcementLearning(
  data,
  s = "State", a = "Action", r = "Reward", s_new = "NextState",
  iter = 1, alpha = 0.1, gamma = 0.9, epsilon = 0.5  # Start with initial epsilon
)

# Storage for epsilon values and rewards
epsilon_values <- numeric(5000)
reward_history <- numeric(5000)

# Iterative training loop
for (i in 1:5000) {
  epsilon_values[i] <- adaptive_epsilon(i, reward_history)
  
  # Update the model instead of retraining
  model <- ReinforcementLearning(
    data,
    s = "State", a = "Action", r = "Reward", s_new = "NextState",
    iter = 1, alpha = 0.1, gamma = 0.9, epsilon = epsilon_values[i],
    model = model  # Retain previous learning
  )
  
  # Store total reward for tracking improvements
  reward_history[i] <- sum(model$Q, na.rm = TRUE)  # Sum Q-values as a proxy for learning progression
}

# Plot reward evolution
ggplot(data.frame(Iteration = 1:5000, Reward = reward_history), aes(x = Iteration, y = Reward)) +
  geom_line(color = "blue") +
  labs(title = "Reward Evolution Over Training Iterations", x = "Iteration", y = "Total Q-value Sum") +
  theme_minimal()

print(model$Q)

# Test policy on new states
test_states <- unique(data$State)[1:5]
predicted_actions <- sapply(test_states, function(s) predict(model, s))
data.frame(State = test_states, RecommendedAction = predicted_actions)

new_states <- sample(unique(data$State), 10)  # Select 10 random new states
predicted_actions <- sapply(new_states, function(s) predict(model, s))
data.frame(State = new_states, RecommendedAction = predicted_actions)