# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")

    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
        """Olympique de Marseille - History and Legacy: This document is about the French football club OM and its history. Olympique de Marseille, founded in 1899, stands as France's most successful club in European competition and the only French team to win the Champions League. Their historic triumph came in 1993 when they defeated AC Milan 1-0 at Munich's Olympiastadion, with Basile Boli scoring the decisive goal.
The club's golden era spanned the late 1980s and early 1990s under president Bernard Tapie, who assembled a star-studded squad including Jean-Pierre Papin, Didier Deschamps, and Chris Waddle. However, this period was later tainted by the match-fixing scandal that led to their relegation to Ligue 2 in 1994.
OM has won Ligue 1 ten times, with their most recent title in 2010. The club's motto "Droit au But" (Straight to Goal) reflects their attacking philosophy. Despite financial struggles in recent decades, Marseille remains France's second-most supported club after Paris Saint-Germain, with a passionate fanbase known as the "Dodgers" who create an intimidating atmosphere at the Stade Vélodrome.""",
"""Le Classique and Historic Rivalries: This document is about Olympique the Marseille and their biggest rivals. Olympique de Marseille's recent transfer activity reflects their financial constraints and strategic player development approach. Between 2020-2024, OM generated approximately €180 million in player sales while spending €120 million on acquisitions, maintaining a positive transfer balance crucial for Financial Fair Play compliance.
Major departures include Boubacar Kamara to Aston Villa (€15 million, 2022), Morgan Sanson to Aston Villa (€16 million, 2021), and Duje Ćaleta-Car to Southampton (€12 million, 2023). These sales demonstrate OM's ability to develop talent and capitalize on market value increases.
Significant acquisitions include Alexis Sánchez (free transfer, 2022), Mattéo Guendouzi (€11 million from Arsenal, 2022), and Jonathan Clauss (€7.5 million from RC Lens, 2022). The club's strategy emphasizes free transfers and loan deals, with approximately 40% of recent signings arriving without transfer fees.
Academy graduates represent crucial assets, with players like Boubacar Kamara and Bamba Dieng generating pure profit when sold. OM typically maintains 6-8 loanees annually, developing young talents while managing squad costs. The average age of new signings has decreased from 26.8 years (2020) to 24.1 years (2024), reflecting a youth-focused recruitment strategy designed for long-term sustainability.""",
"""Transfer Market Activity and Player Trading Statistics: This document is about the transfer activity of Olympique de Marseille (OM). Olympique de Marseille's recent transfer activity reflects their financial constraints and strategic player development approach. Between 2020-2024, OM generated approximately €180 million in player sales while spending €120 million on acquisitions, maintaining a positive transfer balance crucial for Financial Fair Play compliance.
Major departures include Boubacar Kamara to Aston Villa (€15 million, 2022), Morgan Sanson to Aston Villa (€16 million, 2021), and Duje Ćaleta-Car to Southampton (€12 million, 2023). These sales demonstrate OM's ability to develop talent and capitalize on market value increases.
Significant acquisitions include Alexis Sánchez (free transfer, 2022), Mattéo Guendouzi (€11 million from Arsenal, 2022), and Jonathan Clauss (€7.5 million from RC Lens, 2022). The club's strategy emphasizes free transfers and loan deals, with approximately 40% of recent signings arriving without transfer fees.
Academy graduates represent crucial assets, with players like Boubacar Kamara and Bamba Dieng generating pure profit when sold. OM typically maintains 6-8 loanees annually, developing young talents while managing squad costs. The average age of new signings has decreased from 26.8 years (2020) to 24.1 years (2024), reflecting a youth-focused recruitment strategy designed for long-term sustainability.""",
"""Ligue 1 Analytics and Marseille's Performance Metrics: This document is about the statistics of Olympique de Marseille. Ligue 1 has embraced advanced analytics, with Marseille consistently ranking among the top clubs in several key performance indicators. In recent seasons, OM typically ranks second or third in expected goals (xG), demonstrating their attacking intent despite not always converting chances efficiently.
Marseille's pressing intensity averages among Ligue 1's highest, with approximately 23.5 pressures per possession lost, reflecting their aggressive off-ball approach. The Stade Vélodrome advantage becomes evident in home/away performance splits - OM typically gains 1.8 points per home game versus 1.2 away, among the league's largest differentials.
Set-piece efficiency represents another strength, with Marseille scoring approximately 15-20% of their goals from dead ball situations, largely due to Dimitri Payet's delivery. Defensively, their high line results in fewer clearances per game (18.2) but more interceptions in the attacking third (4.1 per match).
Financial Fair Play considerations impact squad building, with OM's wage-to-revenue ratio hovering around 70%, necessitating careful squad management. Youth development remains crucial, with academy graduates contributing approximately 25% of first-team minutes, highlighting the pathway from La Commanderie training center to professional football.""",
"""Stade Vélodrome - Fortress of the Mediterranean: This document is about the stadium of Olympique de Marseille. The Stade Vélodrome, inaugurated in 1937, serves as one of Europe's most atmospheric stadiums and Marseille's spiritual home. With a current capacity of 67,394 following extensive renovations for Euro 2016, it ranks as France's second-largest stadium after the Stade de France.
The stadium's most iconic feature is the "Virage Sud" (South Stand), where Marseille's most passionate supporters, including the Ultras groups like "South Winners" and "Fanatics," create a wall of sound that intimidates visiting teams. The famous "Aux Armes" chant, adapted from La Marseillaise, echoes throughout the stadium, creating an almost religious atmosphere during important matches.
Architecturally, the Vélodrome underwent a €267 million renovation that transformed it into a modern arena while preserving its intimidating atmosphere. The distinctive undulating roof design reflects the Mediterranean waves, symbolizing Marseille's maritime heritage. The stadium has hosted major international events, including 1998 World Cup matches, Euro 2016 fixtures, and serves as a regular venue for France national team matches, cementing its status as a cathedral of French football."""
    ]
    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    
    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page
st.title(" Ⓜ️ Olympique de Marseille")
st.markdown("### 🏟️ **Ask About OM & Ligue 1**")
st.markdown("### 📊 **Match Analytics & Stats**")
st.markdown("### 🏆 **Historic Moments**")
st.markdown("### ⚽ Welcome to **Marseille Central**!")
st.markdown("*Your personal OM and Ligue 1 knowledge assistant*")
st.markdown("🔵⚪ *Droit au But - Straight to the Goal!* ⚪🔵")
st.markdown("""
<style>
    .om-colors {
        background: linear-gradient(90deg, #009CDA 50%, #FFFFFF 50%);
        height: 8px;
        border-radius: 4px;
        margin: 10px 0;
        border: 1px solid #009CDA;
    }
    .om-banner {
        background: linear-gradient(135deg, #009CDA, #0080B8);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        border: 2px solid #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Add the OM color stripe
st.markdown('<div class="om-colors"></div>', unsafe_allow_html=True)

# Optional: Add a styled banner with OM motto
st.markdown("""
<div class="om-banner">
    <strong>🔵⚪ DROIT AU BUT ⚪🔵</strong><br>
    <em>Olympique de Marseille - Since 1899</em>
</div>
""", unsafe_allow_html=True)

# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
st.write("Welcome to the Olympique de Marseille database! Ask me anything about the club's history, players, transfers or statistics.")

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
question = st.text_input("What would you like to know about Olympique de Marseille?",)

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
if st.button("⚽ Find OM Answer", type="primary"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("⚽ Searching the Vélodrome archives..."): 
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**Answer:**")
        st.write(answer)
        st.success("🏆 Found the perfect OM answer for you!")
        #additional message:
        st.warning("🔵 Remember: Allez l'OM! 🔵")
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("Please enter a question!")

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
# OM blue styling for expander
st.markdown("""
<style>
    .streamlit-expanderHeader {
        background-color: #009CDA !important;
        color: white !important;
        border-radius: 5px !important;
    }
    .streamlit-expanderContent {
        background-color: #f0f8ff !important;
        border: 2px solid #009CDA !important;
        border-radius: 0 0 5px 5px !important;
    }
</style>
""", unsafe_allow_html=True)

with st.expander("📚 About this Marseille & Ligue 1 Q&A System"):
    st.markdown("""
    <div style="color: #009CDA; font-weight: bold;">
    I created this system with knowledge about:
    </div>
    <ul style="color: #0080B8;">
    <li>Olympique de Marseille history and Champions League victory</li>
    <li>Le Classique rivalry with PSG and other derbies</li>
    <li>Stade Vélodrome atmosphere and renovations</li>
    <li>Recent transfer market activity and player statistics</li>
    <li>Ligue 1 analytics and performance metrics</li>
    </ul>
    <div style="color: #009CDA; font-style: italic; margin-top: 10px;">
    🔵⚪ Try asking about OM's legendary players, historic matches, or current squad! ⚪🔵
    </div>
    """, unsafe_allow_html=True)

#Additional visual customizations
st.sidebar.markdown("### 🔵 OM Quick Facts")
st.sidebar.info("🏆 Champions League: 1993")
st.sidebar.info("🏟️ Stadium: Stade Vélodrome")
st.sidebar.info("👥 Capacity: 67,394")



# TO RUN: Save as app.py, then type: streamlit run app.py