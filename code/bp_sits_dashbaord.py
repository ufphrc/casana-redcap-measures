import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Define client's color palette
client_colors = ['#6c6df9', '#e5a752', '#286e65', '#bcda45', '#99c1db', '#cdb7c7']

# Load data
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Process data
    def extract_sits(row):
        # Get the total number of sits for the participant
        good_readings = list(map(int, row['good_readings'].split('; ')))
        num_sits = max(good_readings)
        
        # Get the signal strengths
        good_signals = row.filter(like='ths_pf_').tolist()
        poor_sits = good_signals.count('Poor')
        good_sits = good_signals.count('Good')
        
        # Extract BP readings
        readings = [(row[f'sys_{i}'], row[f'dia_{i}']) for i in good_readings]
        return num_sits, poor_sits, good_sits, readings, good_signals

    data[['num_sits', 'poor_sits', 'good_sits', 'readings', 'good_signals']] = data.apply(extract_sits, axis=1, result_type="expand")
    
    # Calculate statistics
    total_participants = len(data)
    cumulative_total_sits = data['num_sits'].sum()
    total_good_sits = data['good_sits'].sum()
    total_poor_sits = data['poor_sits'].sum()
    avg_total_sits = data['num_sits'].mean()
    avg_poor_sits = data['poor_sits'].mean()
    good_sits_percentage = (total_good_sits / cumulative_total_sits) * 100
    poor_sits_percentage = (total_poor_sits / cumulative_total_sits) * 100

    # Extra sits due to BP readings not within the required range
    def calculate_extra_sits(row):
        good_signals = row.filter(like='ths_pf_').tolist()
        good_sits = good_signals.count('Good')
        # Calculate extra sits as the difference between the number of 'Good' signals and 3
        extra_sits = max(0, good_sits - 3)
        return extra_sits

    data['extra_sits'] = data.apply(calculate_extra_sits, axis=1)
    avg_extra_sits = data['extra_sits'].mean()

    # Plotting functions
    def plot_bar_chart(series, title, xlabel, ylabel, definition):
        plt.figure(figsize=(10, 6))
        counts = series.value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.values, palette=client_colors)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(0, counts.max() * 1.2)  # 20% greater than the highest value
        for index, value in enumerate(counts):
            plt.text(index, value + 0.05, f'{value} ({value / len(series) * 100:.2f}%)', ha='center', color=client_colors[1])
        st.pyplot(plt)
        st.write(definition)

    # Displaying results
    st.title('BP Device Validation Dashboard')

    st.header('Overall Statistics')
    st.write(f"**Total Number of Participants:** {total_participants}")
    st.write(f"**Cumulative Number of Total Sits:** {cumulative_total_sits}")
    st.write(f"**Total Good Sits:** {total_good_sits} ({good_sits_percentage:.2f}%)")
    st.write(f"**Total Poor Sits:** {total_poor_sits} ({poor_sits_percentage:.2f}%)")
    st.write(f"**Average Number of Total Sits:** {avg_total_sits:.2f}")
    st.write(f"**Average Number of Poor Sits:** {avg_poor_sits:.2f}")
    st.write(f"**Average Number of Extra Sits Due to BP Readings:** {avg_extra_sits:.2f}")

    st.header('Bar Graphs')
    st.subheader('Frequency of Total Sits')
    plot_bar_chart(
        data['num_sits'], 
        'Frequency of Total Sits', 
        'Number of Sits', 
        'Frequency',
        '**Definition:** Total sits refers to the total number of sits required for each participant to achieve three acceptable BP readings with good signals.'
    )

    st.subheader('Frequency of Poor Sits')
    plot_bar_chart(
        data['poor_sits'], 
        'Frequency of Poor Sits', 
        'Number of Poor Sits', 
        'Frequency',
        '**Definition:**  Poor sits refer to the number of sits where the signal strength was marked as "Poor".'
    )

    st.subheader('Frequency of Extra Sits Due to BP Readings')
    plot_bar_chart(
        data['extra_sits'], 
        'Frequency of Extra Sits Due to BP Readings', 
        'Number of Extra Sits', 
        'Frequency',
        '**Definition:**  Extra sits refer to the number of additional sits required because the BP readings were not within the required range, despite having good signal strength.'
    )
