# Proposal
Being in a large crowd increases the chance of catching, and subsequently spreading, infectious diseases. 
For animals with swarming tendencies, this is a constant risk. When self-isolation isn’t available, what prevents disease from ravaging herds? 

# Model Description
This is a well-mixed compartmental model which tracks epidemic dynamics. It's built to be applicable to many kinds of situations, such as wolves and elk or penguins and polar bears. The "disease" itself can also represent illness or parasites, as long as it has a direct mode of transmission through contact with flock members. \

[The animated model](SIRAnimButtons.py) is built on the previous, simpler models. The user can input a custom epidemic or select one of the [presets](ModelPresets.py) and watch how the epidemic plays out. Inputs include disease strength, recovery rate, mortality rate, predator selection rate, and more.  
