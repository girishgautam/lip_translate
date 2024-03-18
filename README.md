# LipNet: End-to-End Sentence-level Lipreading

This repository contains a Keras implementation of the method described in the paper 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas ([Read the paper](https://arxiv.org/abs/1611.01599)).

## Dataset
The model utilizes the GRID corpus ([GRID corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/)).

## Project Overview
This project was developed as the final capstone project for a Data Science boot camp by a team of four: Alessia S, Ecem K, Mathilda W, and Girish J. The project spanned over 2 weeks, during which we implemented and fine-tuned the LipNet model. We also created a Streamlit app to showcase our model's performance and findings.

## Streamlit App
Explore our lip reading model in action through our Streamlit app: [Lip Reader Tester](https://lip-reader-tester.streamlit.app/).

## Model Architecture & Data Preprocessing
We preprocessed the videos using the dlib library to extract lips from the videos. Text data was tokenized to feed into our model. The core of our approach lies in a deep neural network architecture tailored for lipreading tasks.

## Getting Started
To get started with the LipNet model, follow these steps:
1. Clone this repository: `git clone https://github.com/girishgauta/lip_translate.git`
2. Install the necessary dependencies: `pip install -r requirements.txt`
3. Preprocess your own data or utilize the provided scripts.
4. Train the LipNet model using your data.
5. Evaluate the model's performance and fine-tune as necessary.

## Contributions
Contributions to this project are welcome! Feel free to open issues for bug fixes or enhancements. Additionally, pull requests are encouraged for contributions that align with the project's goals.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
We extend our gratitude to Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas for their groundbreaking work on LipNet. Special thanks to the creators of the GRID corpus for providing the dataset used in this project. We would also like to thank Nicholas Renotte (@nicknochnack) whose YouTube video tutorials were invaluable in helping us understand and structure our own project.
