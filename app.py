import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# Set the page title and layout
st.set_page_config(page_title="DOE-Based Feature Selection", layout="wide")

# Title of the app
st.title('DOE-Based Feature Selection with Minitab Functionalities')

# Sidebar for uploading the dataset
st.sidebar.header('Upload Dataset')
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    st.sidebar.header('DOE Configuration')
    design_type = st.sidebar.selectbox('Select DOE Type', ['Full Factorial', 'Taguchi', 'Fractional Factorial'])

    factors = st.sidebar.multiselect('Select Factors', options=data.columns)
    levels = st.sidebar.number_input('Number of Levels', min_value=2, max_value=10, value=2)

    def full_factorial_design(factors, levels):
        design = list(itertools.product(range(levels), repeat=len(factors)))
        return pd.DataFrame(design, columns=factors)

    def taguchi_design(factors, levels):
        num_runs = len(factors) * levels
        design = np.random.choice(levels, size=(num_runs, len(factors)))
        return pd.DataFrame(design, columns=factors)

    def fractional_factorial_design(factors, levels):
        # Simplified example, in practice, use specific fractional factorial designs
        design = full_factorial_design(factors, levels)
        return design.sample(frac=0.5)

    if design_type == 'Full Factorial':
        design = full_factorial_design(factors, levels)
    elif design_type == 'Taguchi':
        design = taguchi_design(factors, levels)
    else:
        design = fractional_factorial_design(factors, levels)

    st.write("### Design Matrix")
    st.write(design)

    response = st.selectbox('Select Response Variable', options=data.columns)
    if st.button('Analyze'):
        formula = response + ' ~ ' + ' + '.join(factors)
        model = ols(formula, data=data).fit()
        anova_results = anova_lm(model)
        st.write("### ANOVA Results")
        st.write(anova_results)

        def plot_main_effects(design, response):
            plt.figure(figsize=(10, 6))
            for i, factor in enumerate(design.columns):
                plt.subplot(1, len(design.columns), i+1)
                plt.plot(design[factor], data[response], 'o-')
                plt.title(f'Main Effect of {factor}')
                plt.xlabel(factor)
                plt.ylabel(response)
            st.pyplot(plt)

        def plot_interactions(design, response):
            plt.figure(figsize=(10, 6))
            for i, (f1, f2) in enumerate(itertools.combinations(design.columns, 2)):
                plt.subplot(1, len(list(itertools.combinations(design.columns, 2))), i+1)
                plt.plot(design[f1], design[f2], 'o-')
                plt.title(f'Interaction between {f1} and {f2}')
                plt.xlabel(f1)
                plt.ylabel(f2)
            st.pyplot(plt)

        def plot_pareto_effects(anova_results):
            effects = anova_results["sum_sq"]
            effects.sort_values(ascending=False, inplace=True)
            plt.figure(figsize=(10, 6))
            effects.plot(kind='bar')
            plt.title('Pareto Effect Chart')
            plt.xlabel('Factors')
            plt.ylabel('Sum of Squares')
            st.pyplot(plt)

        def plot_partial_effects(model, design, response):
            plt.figure(figsize=(10, 6))
            for factor in design.columns:
                plt.subplot(1, len(design.columns), design.columns.get_loc(factor) + 1)
                pred_vals = model.predict(pd.DataFrame({factor: design[factor]}))
                plt.plot(design[factor], pred_vals, 'o-')
                plt.title(f'Partial Effect of {factor}')
                plt.xlabel(factor)
                plt.ylabel(response)
            st.pyplot(plt)

        if st.button('Show Main Effects'):
            plot_main_effects(design, response)

        if st.button('Show Interactions'):
            plot_interactions(design, response)

        if st.button('Show Pareto Effects'):
            plot_pareto_effects(anova_results)

        if st.button('Show Partial Effects'):
            plot_partial_effects(model, design, response)

        st.header('Conclusion and Recommendations')
        st.write("""
        Based on the DOE analysis, we can conclude the following:
        - Significant factors affecting the response variable.
        - Key interactions between factors.
        - Optimal levels for each factor to achieve the desired outcome.
        """)

        st.download_button(
            label="Download ANOVA Results",
            data=anova_results.to_csv().encode('utf-8'),
            file_name='anova_results.csv',
            mime='text/csv',
        )

else:
    st.write("Please upload a CSV file to proceed.")
