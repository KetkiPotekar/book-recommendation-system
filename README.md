📚 Project Summary: Book Recommendation System Using NLP, Clustering, and Streamlit <br>
🔍 Objective:
To design and deploy an intelligent book recommendation system that provides personalized book suggestions based on user preferences and book metadata. The goal is to enhance the reading experience, improve library/bookstore services, and help authors and publishers understand reader interests.

🧭 Implementation Overview:
Phase 1: Data Preparation
Loaded the dataset Audible_Catalog.csv containing metadata like Book Name, Author, Rating, Number of Reviews, Price, Description, Listening Time, and Genre.

Inspected data structure and saved it in a cleaned CSV format for easy use.

Phase 2: Data Cleaning
Handled missing values and removed duplicate entries.

Converted relevant columns to appropriate data types.

Parsed and cleaned the Ranks and Genre column to extract a primary genre for each book.

Saved the cleaned dataset for modeling and visualization.

Phase 3: Exploratory Data Analysis (EDA)
Analyzed the distribution of ratings, prices, and review counts.

Visualized:

Most common genres

Top-rated authors

Correlation heatmap between Rating, Number of Reviews, and Price

Discovered patterns in reader preferences and genre popularity.

Phase 4: NLP + Clustering
Processed the Description column using NLP:

Tokenization, stopword removal, TF-IDF vectorization

Applied clustering (K-Means) to group books based on content similarity.

Assigned cluster labels to each book to power similarity-based recommendations.

Phase 5: Recommendation Engine Development
Built a Content-Based Filtering model using:

TF-IDF vectors from descriptions

Cosine similarity to find top N similar books

Integrated clustering information to suggest books from the same thematic group.

Developed a hybrid recommendation logic using both description similarity and genre/cluster context.

Phase 6: Streamlit App + AWS Deployment
Designed a user-friendly interface using Streamlit that allows users to:

Search for a book

Get similar book recommendations

Explore genre popularity and top authors

Deployed the app on AWS EC2, making it accessible via a public IP.

🎯 Business Impact & Significance:
✅ Personalized Recommendations:
Readers receive customized suggestions, enhancing their discovery experience based on descriptions and genres.

✅ Enhanced Decision Making:
Bookstores and libraries can optimize shelf space and promotional efforts by analyzing genre trends and ratings.

✅ Data-Driven Insights for Authors/Publishers:
Clusters and trends offer clear feedback on market preferences, helping in targeting content.

✅ Technology Integration:
A full-stack AI-powered application combining:

Python (Pandas, Scikit-learn, NLTK)

NLP (TF-IDF, Cosine Similarity)

Clustering (K-Means)

Streamlit for frontend

AWS EC2 for deployment

🏁 Final Deliverables:
Cleaned and processed book dataset

Multiple recommendation models (content-based, cluster-based)

Streamlit-based app deployed on AWS

EDA visualizations and insights

Evaluation metrics and usability testing

