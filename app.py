import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from scipy import stats

# Set the page title and layout
st.set_page_config(page_title="DOE-Based Feature Selection", layout="wide")

# Title of the app
st.title('DOE-Based Feature Selection with Minitab Functionalities')

# Sidebar for uploading the dataset
st.sidebar.header('Upload Dataset')
uploaded_file = st.sidebar.file_uploader("Choose a file")

try:
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
            try:
                formula = response + ' ~ ' + ' + '.join(factors)
                model = ols(formula, data=data).fit()
                anova_results = anova_lm(model)
                st.write("### ANOVA Results")
                st.write(anova_results)

                def plot_main_effects(design, response):
                    fig, axes = plt.subplots(1, len(design.columns), figsize=(15, 5))
                    for i, factor in enumerate(design.columns):
                        means = data.groupby(factor)[response].mean()
                        axes[i].plot(means.index, means.values, 'o-')
                        axes[i].set_title(f'Main Effect of {factor}')
                        axes[i].set_xlabel(factor)
                        axes[i].set_ylabel(response)
                    st.pyplot(fig)
                    st.write("**Main Effects Plot:** This plot shows the effect of each factor on the response variable. It helps identify which factors have the most significant impact.")

                def plot_interactions(design, response):
                    combinations = list(itertools.combinations(design.columns, 2))
                    fig, axes = plt.subplots(len(combinations), 1, figsize=(10, 5 * len(combinations)))
                    for i, (f1, f2) in enumerate(combinations):
                        means = data.groupby([f1, f2])[response].mean().unstack()
                        means.plot(kind='line', marker='o', ax=axes[i])
                        axes[i].set_title(f'Interaction between {f1} and {f2}')
                        axes[i].set_xlabel(f1)
                        axes[i].set_ylabel(response)
                    st.pyplot(fig)
                    st.write("**Interaction Plot:** This plot shows the interaction between pairs of factors and their combined effect on the response variable.")

                def plot_pareto_effects(anova_results):
                    effects = anova_results["sum_sq"].dropna()
                    effects = effects.astype(float)
                    effects.sort_values(ascending=False, inplace=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    effects.plot(kind='bar', ax=ax)
                    ax.set_title('Pareto Effect Chart')
                    ax.set_xlabel('Factors')
                    ax.set_ylabel('Sum of Squares')
                    st.pyplot(fig)
                    st.write("**Pareto Effect Chart:** This chart shows the sum of squares of each factor. It helps identify the most significant factors affecting the response variable.")

                def plot_partial_effects(model, design, response, factors):
                    fig, axes = plt.subplots(1, len(design.columns), figsize=(15, 5))
                    for i, factor in enumerate(design.columns):
                        try:
                            # Create a DataFrame with all factors set to their mean values
                            pred_data = pd.DataFrame({col: [data[col].mean()] * len(data[factor].unique()) for col in factors})
                            # Vary only the current factor
                            pred_data[factor] = data[factor].unique()
                            
                            # Predict using the model
                            pred_vals = model.predict(pred_data)
                            
                            axes[i].plot(pred_data[factor], pred_vals, 'o-')
                            axes[i].set_title(f'Partial Effect of {factor}')
                            axes[i].set_xlabel(factor)
                            axes[i].set_ylabel(response)
                        except Exception as e:
                            st.error(f"Error in plotting partial effects for {factor}: {e}")
                    st.pyplot(fig)
                    st.write("**Partial Effect Plot:** This plot shows the partial effect of each factor on the response variable. It helps understand the relationship between each factor and the response.")

                def plot_normal_probability_plot(model):
                    residuals = model.resid
                    fig = plt.figure(figsize=(10, 6))
                    stats.probplot(residuals, dist="norm", plot=plt)
                    plt.title('Normal Probability Plot of the Effects')
                    st.pyplot(fig)
                    st.write("**Normal Probability Plot:** This plot shows if the residuals follow a normal distribution. Points should fall roughly along the reference line if the residuals are normally distributed.")

                st.write("### Charts")

                plot_main_effects(design, response)
                plot_interactions(design, response)
                plot_pareto_effects(anova_results)
                plot_partial_effects(model, design, response, factors)
                plot_normal_probability_plot(model)

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
            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")

    else:
        st.write("Please upload a CSV file to proceed.")
except Exception as e:
    st.error(f"An error occurred: {e}")
