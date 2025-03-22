​Top2Vec, an innovative algorithm for topic modeling and semantic search that leverages joint document and word embeddings to identify topics within a corpus without extensive preprocessing or prior knowledge of topic quantity.​

### **Background**

Traditional topic modeling techniques, such as Latent Dirichlet Allocation (LDA) and Probabilistic Latent Semantic Analysis (PLSA), often rely on bag-of-words representations. These methods ignore word order and semantics, necessitating preprocessing steps like stop-word removal, stemming, and lemmatization. Additionally, they typically require the number of topics to be specified beforehand. These limitations can lead to less informative topic representations.​

In contrast, distributed representations, as seen in models like word2vec and doc2vec, capture semantic relationships between words and documents. Building upon these advancements, Top2Vec aims to address the shortcomings of traditional methods by utilizing semantic embeddings for more accurate and automated topic discovery

### **Methodology**

Top2Vec operates through the following key steps:​

1. **Joint Embedding of Documents and Words**: The algorithm begins by creating a semantic space where both document and word vectors are embedded. This is achieved using models like doc2vec, which learn vector representations that capture semantic similarities.​
    
2. **Dimensionality Reduction**: To manage the high-dimensional nature of the embeddings, Top2Vec employs Uniform Manifold Approximation and Projection (UMAP) to reduce the dimensionality, facilitating the identification of dense clusters.​[
    
3. **Clustering**: Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) is then applied to the reduced embeddings to detect dense regions, each representing a potential topic.​
    
4. **Topic Vector Calculation**: For each identified cluster, a centroid is computed in the original high-dimensional space, serving as the topic vector. This vector represents the central theme of the documents within that cluster.​
    
5. **Topic Representation**: The algorithm identifies words whose embeddings are closest to each topic vector, designating them as the representative words for that topic.​

**Advantages**

Top2Vec offers several notable benefits over traditional topic modeling approaches:​

- **Automatic Topic Number Determination**: The algorithm inherently discovers the number of topics present in the corpus without requiring prior specification.​[GitHub+3arXiv+3Semantic Scholar+3](https://ar5iv.labs.arxiv.org/html/2008.09470)
    
- **Minimal Preprocessing**: There is no need for stop-word lists, stemming, or lemmatization, simplifying the preprocessing pipeline.​[GitHub+4Semantic Scholar+4arXiv+4](https://www.semanticscholar.org/paper/Top2Vec%3A-Distributed-Representations-of-Topics-Angelov/fda2a8b03fb15a2d8b5c5aeb01d1c0b27f0b006b)
    
- **Semantic Richness**: By leveraging semantic embeddings, Top2Vec captures nuanced relationships between words and documents, leading to more informative and representative topics.​
    
- **Scalability**: The method is designed to handle large datasets efficiently, making it suitable for extensive corpora.​
    

**Applications**

Top2Vec is versatile and can be applied to various tasks, including:​

- **Topic Discovery**: Uncovering hidden themes within large text collections.​
    
- **Semantic Search**: Enhancing search functionalities by understanding the semantic content of documents.​
  
- **Document Clustering**: Grouping similar documents based on their content for better organization and retrieval.​
    

**Implementation**

An open-source implementation of Top2Vec is available, providing users with tools to apply the algorithm to their datasets. The implementation supports various embedding models, including doc2vec and pre-trained transformers, and offers functionalities for topic exploration, hierarchical topic reduction, and semantic searches.​[GitHub](https://github.com/ddangelov/Top2Vec)[arXiv+1GitHub+1](https://ar5iv.labs.arxiv.org/html/2008.09470)

**Conclusion**

Top2Vec represents a significant advancement in topic modeling by integrating semantic embeddings to automatically and effectively identify topics within a corpus. Its ability to operate with minimal preprocessing and without predefined topic numbers makes it a valuable tool for natural language processing tasks. The algorithm's open-source availability further encourages its adoption and adaptation in various applications.

---

​Top2Vec does not rely solely on word embeddings because its objective is to identify topics based on the semantic content of entire documents, not just individual words. By generating embeddings for both documents and words within the same semantic space, Top2Vec ensures that the context and meaning of each document are captured, allowing for more accurate topic modeling.

---

### **Differences Between Word2Vec, Doc2Vec, and BERT in Generating Representations**

|Feature|**Word2Vec**|**Doc2Vec**|**BERT**|
|---|---|---|---|
|**Representation Type**|Word-level|Document-level|Contextual word-level|
|**Training Approach**|Predicts neighboring words|Predicts neighboring words with document ID|Predicts masked words & next sentence|
|**Context Awareness**|Static (same embedding for a word in all contexts)|Static|Dynamic (different embedding for a word depending on context)|
|**Pretrained on?**|Large text corpus|Large text corpus with document labels|Large text corpus (Bidirectional context learning)|
|**Architecture**|Neural network-based shallow model|Neural network-based shallow model|Transformer-based deep model|
|**Embedding Size**|Fixed (e.g., 300-dim)|Fixed (e.g., 300-dim)|Fixed, but varies across layers (e.g., 768-dim for BERT-base)|
|**Application**|Word similarity, NLP tasks|Document similarity, topic modeling|NLP tasks like QA, translation, classification|

---

## **How They Work**

### **1. Word2Vec: Word Embeddings via Prediction**

Word2Vec is a shallow, two-layer neural network that learns word representations by predicting **word relationships** using two methods:

- **Skip-gram:** Given a word, predict surrounding words.
- **Continuous Bag of Words (CBOW):** Given surrounding words, predict the missing word.

#### **Limitations**

- **Static embeddings**: The same word always has the same vector, regardless of context (e.g., "bank" in "river bank" vs. "bank loan").
- **Only word-level**: No document-level representation.

---

### **2. Doc2Vec: Extending Word2Vec for Document Representations**

Doc2Vec extends Word2Vec to learn representations of **entire documents** by introducing a **document vector** (`D`).

- **PV-DM (Distributed Memory)**: Learns a document vector like a missing word in CBOW.
- **PV-DBOW (Distributed Bag of Words)**: Learns a document vector by predicting words like skip-gram.

Example:

- Given **Document ID** + **some words**, Doc2Vec predicts missing words.
- After training, each document gets a unique vector representation that captures **topic-level semantics**.

#### **Why It’s Useful?**

- Generates **document embeddings** for clustering and similarity search.
- Used in **Top2Vec** for topic modeling.

#### **Limitations**

- Still **static**: A document's embedding does not dynamically change with context.

---

### **3. BERT: Contextual Word Representations Using Transformers**

BERT (**Bidirectional Encoder Representations from Transformers**) generates **context-aware word embeddings** using the **Transformer** model. It is trained using:

1. **Masked Language Model (MLM):** Randomly masks words in a sentence and asks the model to predict them.
2. **Next Sentence Prediction (NSP):** Determines if two sentences logically follow each other.

#### **How BERT Works**

- Instead of predicting surrounding words like Word2Vec, BERT **understands full sentence context bidirectionally**.
- Uses **self-attention** to weigh word relationships dynamically.

Example: For "The **bank** of the river" vs. "I went to the **bank** to withdraw money":

- BERT assigns different vectors to **bank** based on context.

#### **Advantages**

- **Dynamic embeddings**: Words change meaning depending on the sentence.
- **Pretrained on vast text data**: It generalizes better for downstream tasks.

---

## **Final Comparison**

- **Word2Vec & Doc2Vec** = Static embeddings; useful for clustering, topic modeling, word similarity.
- **BERT** = Context-aware; best for NLP tasks like Q&A, translation, and text classification.

---

### **How Doc2Vec Works**

Doc2Vec extends Word2Vec by learning **fixed-length vector representations for entire documents** rather than just words. It does this by introducing a **document ID (vector D)** into the training process, which helps the model learn document-level semantic relationships.

Doc2Vec has two main architectures:

### **1. PV-DM (Distributed Memory)**

This method is similar to Word2Vec’s **CBOW** but includes a **document vector** as input. The model predicts a missing word in a sentence using both the **surrounding words and the document vector (D)**.

#### **Example**

For the sentence:

> "The cat sat on the mat."

- If the missing word is **"mat"**, the model is trained to predict it using:
    
    - The context words **(The, cat, sat, on, the)**
    - The **document vector (D)** (a unique identifier for the document)
- The document vector **learns to encode the topic or meaning of the entire document**.
    

#### **Why PV-DM?**

- It **preserves word order** by including context words.
- The **document vector (D) helps capture long-term dependencies**.


### **2. PV-DBOW (Distributed Bag of Words)**

This method is similar to **Word2Vec’s Skip-gram**, but instead of using words to predict surrounding words, it **uses the document vector (D) alone to predict randomly sampled words from the document**.

#### **Example**

For a document represented by **D**, the model is trained to predict words like:

> ("cat", "sat", "mat")

- The document vector learns **semantic meaning** by optimizing the prediction of its words.
- Unlike PV-DM, **PV-DBOW does not use surrounding words**, making it **simpler but effective for learning document representations**.

#### **Why PV-DBOW?**

- More **computationally efficient** than PV-DM.
- Can work well even without word ordering.


### **Final Training Process**

- The model trains using **stochastic gradient descent (SGD) and backpropagation**.
- Document vectors are optimized like word embeddings and stored in a lookup table.
- Once trained, the document vector **can be used for clustering, similarity search, or classification**.

### **Key Takeaways**

- **PV-DM**: Learns document embeddings by predicting a missing word using both surrounding words and document ID.
- **PV-DBOW**: Learns document embeddings by predicting words from the document using the document ID only.
- After training, the **document vector can represent an entire document’s meaning** in a way that captures **semantic similarity**.

---

### **. Computing the Centroid in the Original High-Dimensional Space**

- Once clusters are identified in the **low-dimensional UMAP space**, Top2Vec **maps the cluster back to the original high-dimensional space**.
    
- The **centroid of each cluster is computed** in this original space using:
    
    C=1N∑i=1NDiC = \frac{1}{N} \sum_{i=1}^{N} D_iC=N1​i=1∑N​Di​
    
    where:
    
    - CCC is the **topic vector** (centroid),
    - NNN is the **number of document vectors** in the cluster,
    - DiD_iDi​ represents each **document embedding** in the cluster.

### **5. Assigning Topic Words**

- The **nearest word vectors** to the centroid in the original high-dimensional space are selected as **topic words**.
- These words **best represent** the meaning of the cluster.

### **Why Compute Centroids in the Original Space?**

- **Dimensionality reduction distorts distances** slightly, so the **most accurate topic vector** should be computed in the original embedding space.