import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import plotly.express as px
import streamlit as st

# ------------------------------
# 1. Load Data (2 Years × 3 Phases)
# ------------------------------

# Load Year 1 data — contains only category-wise cutoff details
year1_df = pd.read_excel("year1_categorywise.xlsx")
year1_df["Year"] = 1  # Add year identifier

# Load Year 2 data — contains detailed college-wise, branch-wise data for 3 phases
year2_dfs = []
for phase_id in range(1, 4):
    df = pd.read_excel(f"year2_phase{phase_id}.xlsx")
    df["Year"] = 2  # Add year identifier
    df["Phase"] = phase_id  # Add phase identifier
    year2_dfs.append(df)

# Combine all 3 phases of Year 2 into a single dataframe
year2_df = pd.concat(year2_dfs, ignore_index=True)

# Merge Year 1 and Year 2 data into a single cutoff DataFrame
cutoff_df = pd.concat([year1_df, year2_df], ignore_index=True)

# Load previous year marks vs rank data to train prediction model
marks_rank_df = pd.read_excel("marks_rank_previous_years.xlsx")

# Remove any missing data (important for ML model)
marks_rank_df.dropna(inplace=True)
cutoff_df.dropna(inplace=True)

# ------------------------------
# 2. Rank Prediction Model (Trained on Previous Years Data)
# ------------------------------

# Create feature (X) and label (y) arrays
X = marks_rank_df[['Marks']].values
y = marks_rank_df['Rank'].values

# Polynomial Regression (degree 3) for better curve-fitting than linear model
poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_model.fit(X, y)  # Train the model

# Function to predict rank based on marks
def predict_rank(marks):
    return int(poly_model.predict([[marks]])[0])

# ------------------------------
# 3. College Predictor (Flexible for both simple and detailed datasets)
# ------------------------------

def college_predictor(rank, category, year=None, phase=None):
    df = cutoff_df.copy()

    # Filter by year if selected
    if year:
        df = df[df['Year'] == year]

    # Filter by phase only if Year 2 and phase info exists
    if phase and 'Phase' in df.columns:
        df = df[df['Phase'] == phase]

    # Filter colleges that accept ranks higher than or equal to predicted rank
    if 'Cutoff Rank' in df.columns:
        df = df[df['Cutoff Rank'] >= rank]

    # Filter by selected category
    if 'Category' in df.columns:
        df = df[df['Category'] == category]

    # Select display columns (dynamically choose based on available data)
    cols = [c for c in ['College Name', 'Branch', 'Cutoff Rank', 'Fees', 'Phase', 'Year'] if c in df.columns]
    
    # Sort results by either Cutoff Rank or first column (e.g., College Name)
    return df[cols].sort_values(by=cols[-2] if 'Cutoff Rank' in cols else cols[0])

# ------------------------------
# 4. Streamlit UI
# ------------------------------

st.title("🎓 EAMCET Rank & College Predictor (2 Years, 3 Phases)")

# User input for EAMCET marks
marks = st.number_input("Enter your EAMCET Marks", 0, 160, 120)

# User input for category
category = st.selectbox("Select Category", cutoff_df['Category'].unique())

# User input for Year (1 or 2)
year = st.selectbox("Select Year", sorted(cutoff_df['Year'].unique()))

# Show phase selection only if Year 2 is selected
phase = None
if year == 2:
    phase = st.selectbox("Select Phase", sorted(cutoff_df['Phase'].dropna().unique()))

# When Predict button is clicked
if st.button("Predict"):
    # Step 1: Predict rank
    rank = predict_rank(marks)
    st.success(f"Predicted Rank for {marks} Marks: {rank}")

    # Step 2: Show eligible colleges for given rank/category/year/phase
    st.subheader("📋 Eligible Colleges")
    colleges = college_predictor(rank, category, year, phase)
    st.dataframe(colleges)

    # Step 3: Display college-specific details (branches, fees, phase cutoffs)
    if 'College Name' in colleges.columns:
        st.subheader("🏫 College Details")
        for college in colleges['College Name'].unique():
            st.markdown(f"### {college}")
            college_data = colleges[colleges['College Name'] == college]

            # Show branches offered
            if 'Branch' in college_data.columns:
                st.write(f"📍 Branches: {', '.join(college_data['Branch'].unique())}")
            
            # Show fees
            if 'Fees' in college_data.columns:
                st.write(f"💰 Fees: ₹{college_data.iloc[0]['Fees']}")
            
            # Show cutoff rank phase-wise
            if 'Phase' in college_data.columns:
                phase_cutoffs = college_data.groupby('Phase')['Cutoff Rank'].min()
                st.write("📊 Phase-wise Lowest Cutoff Ranks:")
                st.table(phase_cutoffs.reset_index())

    # ------------------------------
    # 5. Visualization Section
    # ------------------------------
    st.subheader("📈 Visualization")

    # Line Chart: Marks vs Rank trend (from previous year data)
    fig_line = px.line(
        marks_rank_df,
        x="Marks", y="Rank", color="Year",
        title="Marks vs Rank Trend (Previous Years)"
    )
    st.plotly_chart(fig_line)

    # Bar Chart: Cutoff Rank by College & Branch, split by Year
    if 'College Name' in cutoff_df.columns and 'Cutoff Rank' in cutoff_df.columns:
        bar_fig = px.bar(
            cutoff_df,
            x="College Name", y="Cutoff Rank",
            color="Branch" if 'Branch' in cutoff_df.columns else None,
            facet_col="Year",
            title="College Cutoff Ranks by Year and Phase"
        )
        st.plotly_chart(bar_fig)

    # Box Plot: Category-wise cutoff rank distribution
    if 'Category' in cutoff_df.columns:
        cat_fig = px.box(
            cutoff_df,
            x="Category", y="Cutoff Rank", color="Year",
            title="Category-wise Cutoff Distribution"
        )
        st.plotly_chart(cat_fig)

    # Heatmap: College vs Branch cutoff rank comparison
    if 'Branch' in cutoff_df.columns:
        heatmap_data = cutoff_df.pivot_table(
            values='Cutoff Rank',
            index='College Name',
            columns='Branch',
            aggfunc='min'
        )
        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="College vs Branch Cutoff Heatmap"
        )
        st.plotly_chart(fig_heatmap)
