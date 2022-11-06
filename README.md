# Financial Assistant Chatbot

Edit: Some functionalities are under construction due to expired APIs

Hello! I built this application because I was fascinated by natural processing and neural networks, and I wanted an opportunity to use them myself, in a form of a chatbot. This project is a major expansion of the "Hello World Hackathon" hosted by Purdue University. The project done in the Hackathon was a beginner-friendly stock market website with essential information, such as stock analysis, top stock gainers and losers by percentage (Done with the React framework). However, my financial assistant chatbot with be done using the Python GUI, with a lot more functionalities, such as predicting the stock prices and graphing the data of the stock prices over period of time.

In this repository, there are two ways the run the code. One is by the GUI (app.py), however, I recommend the one that uses terminal (chatbot.py). The program that utilizes the terminal perfectly works as intended. The program intents are stored in the intents json file that I created manually. I used the dictionary format and added subparts such as "tags", "patterns", and "responses". The tags are basically to identify the intent. For example, in this program, you can say, Hi" The chatbot will connect it to the tag "Greetings". This tag chooses the random response that is stored in the dataset that corresponds to the tag.

(P.S. I named my chatbot ELIZA because it is a reference to one of the earliest natural language processing computer program in the world, made in MIT Artifical Intelligence Laboratory by Joeseph Weizenbuam) 


## External Libraries

To use this program, you need to download loads of different external libraries. They are listed below:

 - Requests
 - NLTK
 - Tensorflow
 - SKlearn
 - Pickle
 - MatPlotLib
 - Pandas
 
 There may be more, but to download them, use -pip install {library name}
 
 
 ## End Result
 
The program itself is still under construction. The parts that needs immediate improvements are the design of the GUI, data visualization, context set in the intents json file, and the training accuracy. However, the program still works as intended and is functional. This is the program that uses wide range of my interests, such as machine learning, artificial intelligence, data visualization, and finance. From one project, I have learned numerous valuable skills and conceps such as how natural langauges and neural networks work, visualizing the data from a dataset, extracting a .json file to use for its data, and using GUI. This project is by far my most valuable work, and I am very proud of myself!
 
 
