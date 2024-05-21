import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

# Define client's color palette
client_colors = ['#6c6df9', '#e5a752', '#286e65', '#bcda45', '#99c1db', '#cdb7c7']

# Streamlit app
st.title("Recruitment Data Visualization")

# Upload the Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    # Load the 'screening' sheet content into a DataFrame
    df = pd.read_excel(uploaded_file, sheet_name='screening')

    # Set the first row as the new header
    df.columns = df.iloc[0]
    df = df[1:]
    # Remove the first row
    df = df.iloc[1:]

    # Remove rows where only 'redcap_consent_id' or 'redcap_visit_id' are filled and no other columns
    df = df.dropna(how='all', subset=df.columns.difference(['redcap_consent_id', 'redcap_visit_id']))
    
    # Remove any completely empty rows
    df = df.dropna(how='all')

    # Replace null values with "Unknown" in relevant columns
    df['type_consent'] = df['type_consent'].fillna('Unknown')
    df['econsent_signed_part'] = df['econsent_signed_part'].fillna('No')
    df['gender'] = df['gender'].fillna('Unknown')
    df['coordinator'] = df['coordinator'].fillna('Unknown')
    df['ph_screening_eligible'] = df['ph_screening_eligible'].fillna('Unknown')
    df['econsent_signed_coord'] = df['econsent_signed_coord'].fillna('No')

    # Strip leading/trailing spaces from gender values
    df['gender'] = df['gender'].str.strip()

    # Function to add annotations on top of the bars
    def add_annotations(ax, data):
        for p in ax.patches:
            height = p.get_height()
            percentage = height / data.sum() * 100
            ax.annotate(f'{height:.0f}\n({percentage:.1f}%)',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 9),
                        textcoords='offset points')

    # 1. # and % e-consent vs physical consent
    st.subheader("Number of participants (%) agreed to: e-consent vs Physical Consent")
    consent_counts = df['type_consent'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=consent_counts.index, y=consent_counts.values, ax=ax, palette=client_colors[:2])
    ax.set_ylim(0, consent_counts.max() * 1.2)  # Increase y-axis by 20%
    ax.set_title('Number of participants (%) agreed to: e-consent vs Physical Consent')
    ax.set_xlabel('Consent Type')
    ax.set_ylabel('Count')
    add_annotations(ax, consent_counts)
    total_count = consent_counts.sum()
    ax.annotate(f'Total: {total_count}', xy=(1, 1.05), xycoords='axes fraction', ha='right', fontsize=12, weight='bold')
    st.pyplot(fig)

    # 2. Out of e-consenting agreed people, how much % of e-consents signed
    st.subheader("Number of e-consents Signed Out of e-consenting Agreed People")
    econsent_agreed = df[df['type_consent'] == 'eConsent']
    econsent_signed = econsent_agreed['econsent_signed_part'].value_counts()

    if not econsent_signed.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=econsent_signed.index, y=econsent_signed.values, ax=ax, palette=client_colors[:2])
        ax.set_ylim(0, econsent_signed.max() * 1.2)  # Increase y-axis by 20%
        ax.set_title('Number of e-consents Signed Out of e-consenting Agreed People')
        ax.set_xlabel('eConsent Signed')
        ax.set_ylabel('Count')
        add_annotations(ax, econsent_signed)
        total_count = econsent_signed.sum()
        ax.annotate(f'Total: {total_count}', xy=(1, 1.05), xycoords='axes fraction', ha='right', fontsize=12, weight='bold')
        st.pyplot(fig)
    else:
        st.write("No data available for e-consenting agreed people.")

    # 3. # and % of males vs females signed up
    st.subheader("Gender Distribution")
    gender_counts = df['gender'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax, palette=client_colors[:3])
    ax.set_ylim(0, gender_counts.max() * 1.2)  # Increase y-axis by 20%
    ax.set_title('Number of Males and Females')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    add_annotations(ax, gender_counts)
    total_count = gender_counts.sum()
    ax.annotate(f'Total: {total_count}', xy=(1, 1.05), xycoords='axes fraction', ha='right', fontsize=12, weight='bold')
    st.pyplot(fig)

    # 4. Bar chart of individuals signed up by each coordinator
    st.subheader("Number of Individuals signed up by each Coordinator")
    coordinator_counts = df['coordinator'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=coordinator_counts.index, y=coordinator_counts.values, ax=ax, palette=client_colors[:len(coordinator_counts)])
    ax.set_ylim(0, coordinator_counts.max() * 1.2)  # Increase y-axis by 20%
    ax.set_title('Number of Individuals signed up by each Coordinator')
    ax.set_xlabel('Coordinator')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    add_annotations(ax, coordinator_counts)
    total_count = coordinator_counts.sum()
    ax.annotate(f'Total: {total_count}', xy=(1, 1.05), xycoords='axes fraction', ha='right', fontsize=12, weight='bold')
    st.pyplot(fig)

    # Additional Visualizations for Tracking Recruitment

    # Distribution of age
    st.subheader("Distribution of Age")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['age'].dropna(), bins=10, kde=True, ax=ax, color=client_colors[0])
    ax.set_title('Distribution of Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    total_count = df['age'].dropna().count()
    ax.annotate(f'Total: {total_count}', xy=(1, 1.05), xycoords='axes fraction', ha='right', fontsize=12, weight='bold')
    st.pyplot(fig)

    # ph_screening_eligible (bar chart)
    st.subheader("Number of participants eligible in Phone Screening")
    ph_screening_counts = df['ph_screening_eligible'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=ph_screening_counts.index, y=ph_screening_counts.values, ax=ax, palette=client_colors[:2])
    ax.set_ylim(0, ph_screening_counts.max() * 1.2)  # Increase y-axis by 20%
    ax.set_title('Number of participants eligible in Phone Screening')
    ax.set_xlabel('Eligibility')
    ax.set_ylabel('Count')
    add_annotations(ax, ph_screening_counts)
    total_count = ph_screening_counts.sum()
    ax.annotate(f'Total: {total_count}', xy=(1, 1.05), xycoords='axes fraction', ha='right', fontsize=12, weight='bold')
    st.pyplot(fig)

    # econsent_signed_coord (Yes/No)
    st.subheader("eConsent signed by Coordinator or not?")
    econsent_signed_coord_counts = df['econsent_signed_coord'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=econsent_signed_coord_counts.index, y=econsent_signed_coord_counts.values, ax=ax, palette=client_colors[:2])
    ax.set_ylim(0, econsent_signed_coord_counts.max() * 1.2)  # Increase y-axis by 20%
    ax.set_title('econsent_signed_coord')
    ax.set_xlabel('eConsent Signed by Coordinator')
    ax.set_ylabel('Count')
    add_annotations(ax, econsent_signed_coord_counts)
    total_count = econsent_signed_coord_counts.sum()
    ax.annotate(f'Total: {total_count}', xy=(1, 1.05), xycoords='axes fraction', ha='right', fontsize=12, weight='bold')
    st.pyplot(fig)

    # Find coordinators with unsigned consent forms
    st.subheader("Select a Coordinator to View Unsigned Consent Forms")
    unsigned_consents = df[df['econsent_signed_coord'] == 'No']
    unsigned_consents_count = unsigned_consents.groupby('coordinator').size().reset_index(name='Unsigned Consent Count')

    # Create radio buttons for coordinators
    coordinator_options = [f"{row['coordinator']} ({row['Unsigned Consent Count']})" for _, row in unsigned_consents_count.iterrows()]
    selected_coordinator = st.radio("Coordinators", coordinator_options)

    # Extract coordinator name from the selected option
    selected_coordinator_name = selected_coordinator.split(" (")[0]

    # Display details for the selected coordinator
    st.subheader(f"Details for {selected_coordinator_name}")
    coordinator_details = unsigned_consents[unsigned_consents['coordinator'] == selected_coordinator_name][['redcap_consent_id', 'econsent_signed_part']]
    st.write(coordinator_details)
