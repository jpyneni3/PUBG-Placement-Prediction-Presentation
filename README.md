# PUBG Placement Prediction
## CS 7641 Team 6: Jas Pyneni, Hemanth Chittanuru, Kavin Krishnan, Vishal Vijaykumar, Rafael Hanashiro

---
<p align="center">
  <img src="https://storage.googleapis.com/kaggle-media/competitions/PUBG/PUBG%20Inlay.jpg">
</p>

# 1. Overview
PUBG is a battle-royale video game where 100 players battle on a game map by rummaging for weapons and tools and fighting until there is only one surviving player or team. Inititally a Kaggle competition, the purpose of this project is to predict the final placement of a given player based off of game stats. This is an interesting project for our group because it is a game we all love to play. We hope to utilize the data modeling skills we learned this semester in CS 7641 to optimize our chance of winning our future matches.

We aim to do this through the following steps:
1. Explore the data
2. Pre-process and engineer the features
3. Model the data and build a prediction engine

# 2. Data Exploration

### Dataset: PUBG Match Placement Data
#### Collected from [Kaggle](https://www.kaggle.com/c/pubg-finish-placement-prediction/data)
#### 4446966 starting stats, 28 features, 1 target
#### Features:
1. Id - unique identifier of player
2. groupId - ID to identify a group within a match.
3. matchId - ID to identify match
4. assists -  Number of enemy players this player damaged that were killed by teammates.
5. boosts - Number of boost items used.
6. damageDealt - Total damage dealt - self inflicted damage
7. DBNOs - Number of enemy players knocked.
8. headshotKills  -  Number of enemy players killed with headshots
9. heals - Number of healing items used.
10. killPlace - Ranking in match of number of enemy players killed.
11. killPointsElo - Kills-based external ranking of player. (
12. kills - Number of enemy players killed.
13. killStreaks - Max number of enemy players killed in a short amount of time.
14. longestKill - Longest distance between player and player killed at time of death.
15. matchDuration - Duration of match in seconds.
16. matchType - String identifying the game mode that the data comes from.
17. maxPlace -  Worst placement we have data for in the match.
18. numGroups - Number of groups we have data for in the match.
19. rankPointsElo  - Elo-like ranking of player.
20. revives - Number of times this player revived teammates.
21. rideDistance - Total distance traveled in vehicles measured in meters.
22. roadKills - Number of kills while in a vehicle.
23. swimDistance - Total distance traveled by swimming measured in meters.
24. teamKills -  Number of times this player killed a teammate.
25. vehicleDestroys - Number of vehicles destroyed.
26. walkDistance - Total distance traveled on foot measured in meters.
27. weaponsAcquired -  Number of weapons picked up.
28. winPoints -  Win-based external ranking of player.

#### Target:
1. winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match.

#### Feature Characteristics:
1. int IDs to identify players, groups and matches
2. Discrete data about stats (assists, boosts, headshotKills)
3. External rankings  (based off of some post-match math): ELO scores
4. Certain features listed as inconsistent (rankPointsElo)

### Visualization

#### Feature Analysis
A visualization to show how each individual feature is correlated with the other features

<p align="center">
  <img src="https://raw.githubusercontent.com/jpyneni3/PUBG-Placement-Prediction-Presentation/master/Images/heatmap.jpeg" width="500"/>
</p>

The last column in the above matrix is the most important as it shows the correlation of each feature with our target variable winPlacePerc.


From the heat map we can see that there is a high positive correlation between walkDistance and winPlacePerc. This is interesting and so let's explore the relationship of all types of traveling totalDistance = (walkDistance, rideDistance and swimDistance) and winPlacePerc.

<p align="center">
  <img src="https://raw.githubusercontent.com/jpyneni3/PUBG-Placement-Prediction-Presentation/master/Images/travel_heatmap.jpeg" width="500"/>
</p>

We can see that there is high positive correlation between totalDistance and winPlacePerc as we saw for walkDistance and winPlacePerc.

Let's visualize the relation between walkDistance and winPlacePerc as well as totalDistance and winPlacePerc.

<p align="center">
  <img src="https://raw.githubusercontent.com/jpyneni3/PUBG-Placement-Prediction-Presentation/master/Images/walk_to_win.jpeg" width="500"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/jpyneni3/PUBG-Placement-Prediction-Presentation/master/Images/travel_to_win.jpeg" width="500"/>
</p>

#### Scatter plots of Relations


# 3. Pre-processing

# 4. Modeling, Experimenting & Results

### a. XGBoost

### b. Neural Network

#### Architecture
We used a feedforward neural network with an input layer (fully connected to first hidden layer), 2 fully connected hidden layers and and an output layer to model the data and make a prediction for final PUBG placement. Below is an overview of the architecture used:
<p align="center">
  <img src="https://raw.githubusercontent.com/jpyneni3/PUBG-Placement-Prediction-Presentation/master/Images/NN_architecture.jpeg" width="500"/>
</p>

In the architecture, the size of the input layer is equal to the number of features in the input (size of the input vector) and the output layer has size 1 since we are outputting a single placement prediction. However, the size of the hidden layers is an important hyperparameter in neural networks that can be tuned. We experimented with different hidden layer sizes. In addition, we also ran extensive tests to determine optimal values for several other hyperparameters. Below we present an explanation of our approach for tuning different hyperparameters and the results used to determine optimal values.

#### Hyperparameter Tuning

### Number of Hidden Units
The first hyperparameter we tuned is the number of hidden units (size of the hidden layers). We used all powers of 2 in the range [2, 256] as the hidden unit size and modeled the data using the resulting neural networ. We measured the Mean Absolute Error (MAE) for each configuration and determined that the optimal value is achieved when there are 32 hidden units in the hidden layers. Below is a plot of the results:
<p align="center">
  <img src="https://raw.githubusercontent.com/jpyneni3/PUBG-Placement-Prediction-Presentation/master/Images/NN_hidden_units.jpeg" width="500"/>
</p>

We note above that when the number of hidden units is too small, the model is not complex enough to capture the patterns in the data, thus leading to underfitting and high prediction error. Similarly, when the number of hidden units is too large, the model overfits the data which also produces high prediction error. Thus 32 seems to be the optimal value for the number of hidden units and the error is minimized.

### Learning Rate
Another important hyperparameter in neural networks is the learning rate. The learning rate determines the step size along the gradient of the loss function at each iteration and as such impacts convergence. If the step size is too large, it is possible that we step over the minima and never reach it. If the step size is too small, then convergence might take extremely long and require multiple training iterations. When, the learning error was compared to the training loss, an optimal value of ~0.1 was observed as seen below:
<p align="center">
  <img src="https://raw.githubusercontent.com/jpyneni3/PUBG-Placement-Prediction-Presentation/master/Images/NN_lr.jpeg" width="500"/>
</p>

### Optimizer
