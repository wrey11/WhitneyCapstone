# Exploring Machine Learning as a Solution to Predictive Maintenance in Lean Manufacturing
### Introduction
<br />
In recent years, there have been many philosophies proposed to increase efficiency and
reduce waste in a manufacturing environment. Lean manufacturing is a popular approach,
deriving principles and methods from the famous “Toyota Way” (Marksberry, 2011). All
elements of lean 6-sigma are based on the foundation of Kaizen, or continuous improvement.
One protruding methodology from lean manufacturing is total productive maintenance (TPM).
TPM practices aim to achieve optimal equipment and operational efficiency by assessing and
monitoring people’s behavior, equipment, and production processes (Venkatesh, 2015). TPM is
constructed of eight pillars: autonomous maintenance, planned maintenance, quality
maintenance, focused improvement, early equipment management, training, safety, and
administration (Lean Factories, 2020). Preventative maintenance strategies are often used to
address planned maintenance, focused improvement, and early equipment management pillars.

This project focuses on exploration of using predictive classifiers to identify manufacturing machine failures based on various environmental factors and process parameters. It utilizes existing open-source data from UCI's data repository accessed at: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset.
<br />
<br />
### Approach to Implementation 
The primary objective of this project is to explore various classifier models, ultimately providing the best recommendation for a binary classification of machine failure given multivariate input. 
The program was initially developed, tested, debugged, and locally deployed using Jupyter Notebook - this allowed for initial proof of concept.
Once complete, the code was ported to Pycharm, in which it was adapted for web-based deployment, constructing a user-interactive web application via Streamlit.
The application was tested locally, then publicly deployed using this GitHub repository in conjunction with Streamlit.
<br />
<br />
Both the original code intended for local deployment and the code for the Streamlit application are available within this repository.

### Running the Code
#### Local Deployment
To run this locally:
<br />
   1. Download data file "aiai2020.csv" available in Github repository "WhitneyCapstone".
   2. Take note of data file local path.
   3. Download Python file "Predictive Maintenance-LocalFile (1).py".
   4. Open file in IDE of choice. It was originally written in Juypter Notebook, built in blocks, with a verification activty to ensure each block carried out designed function(s). Each block of code is represented by a "# ln[]".
   5. Update lines 13 and 14 to include local path to downloaded dataset.
   6. Run code! 
#### Web Application 
To access the deployed application please use the following link: https://whitneygcucapstone.streamlit.app/
<br /> Though the development team currently does not plan to provide updates to this application, updates will be automatically ported over to the Streamlit app as a result of revision change within the associated Github file.

