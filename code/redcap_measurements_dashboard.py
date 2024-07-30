import pandas as pd
import re
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Prompt the user to upload the CSV file from REDCap
st.subheader("Upload the CSV file exported from REDCap")

st.markdown("""
Please export the CSV (raw data) from REDCap using the following steps:
1. Go to the REDCap Data Exports, Reports, and Stats column under the Applications section.
2. In "All data (all records and fields)", click the "Export" button.
3. Choose the "**CSV (raw data)**" option.
4. **Do not** click the checkbox for "Remove All Identifier Fields (tagged in Data Dictionary)".
5. Click "Export Data".
6. Upload the exported CSV file below (**DON'T CONSIDER THE NameError, UPLOAD THE FILE**).
""")

# Upload the CSV file
redcap_file = st.file_uploader("Choose a REDCap CSV file", type="csv")

# Prompt the user to upload the second CSV file
st.subheader("Upload the second CSV file")

st.markdown("""
Please upload the second CSV file which contains the columns 'rec_id' and 'good_readings'.
""")

# Upload the second CSV file
second_file = st.file_uploader("Choose the second CSV file", type="csv")

if redcap_file is not None and second_file is not None:
    # Load the uploaded files into DataFrames
    df_redcap = pd.read_csv(redcap_file)
    df_second = pd.read_csv(second_file)
    
    # User inputs for date range
    start_date = st.date_input("From Date", value=pd.to_datetime('2024-01-01'))
    end_date = st.date_input("Until Date", value=pd.to_datetime('2024-12-31'))

    # Function to filter the DataFrame based on user-provided date range
    def filter_by_date(df, date_column, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        mask = (pd.to_datetime(df[date_column]) >= start_date) & (pd.to_datetime(df[date_column]) < end_date)
        return df[mask]

    # Filter the REDCap DataFrame based on the selected date range
    df_redcap = filter_by_date(df_redcap, 'demo_screening_date', start_date, end_date)
    
    # Filter the second DataFrame based on the selected date range
    df_second = filter_by_date(df_second, 'time_1', start_date, end_date)
    
    # Match rec_id with record_id in df_redcap
    df_redcap = df_redcap[df_redcap['record_id'].isin(df_second['rec_id'])]

    # Add radio buttons for filtering options
    filter_option = st.radio("Select option:", ('Only good readings', 'All'))

    # Filter based on the selected option
    if filter_option == 'Only good readings':
        df_second = df_second[df_second['good_readings'].notnull()]
        df_redcap = df_redcap[df_redcap['record_id'].isin(df_second['rec_id'])]

    # The final DataFrame after all filters
    df = df_redcap
    #------------------------------------------------------------------------------------------------------------------------------------#
    # List of columns to delete by serial number including timestamps
    columns_to_delete = [
        'redcap_survey_identifier', 'demo_screening_date', 'demographics_timestamp', 'demo_disability___0', 'demo_disability___1', 
        'demo_disability___2', 'demo_disability___3', 'demo_disability___4', 'demo_disability___5', 
        'demo_disability___6', 'demo_disability_spec', 'demo_lived_in_villages', 'demo_residency', 
        'demo_years_lived_villages', 'demo_surrounding', 'medical_history_timestamp', 'demo_fam_med___1', 
        'demo_fam_med___2', 'demo_fam_med___3', 'demo_fam_med___4', 'demo_fam_med___5', 'demo_fam_med___6', 
        'demo_fam_med_other', 'med_his_diabetes', 'med_his_dia_medic___1', 'med_his_dia_medic___2', 
        'med_his_dia_medic___3', 'med_his_dia_medic___4', 'med_his_dia_medic___5', 'med_his_dia_other', 
        'med_his_bp', 'med_his_bp_medic___1', 'med_his_bp_medic___2', 'med_his_bp_medic___3', 'med_his_bp_medic___4', 
        'med_his_bp_medic___5', 'med_his_bp_other', 'med_his_cardio', 'med_his_cardio_medic___1', 
        'med_his_cardio_medic___2', 'med_his_cardio_medic___3', 'med_his_cardio_medic___4', 'med_his_cardio_medic___5', 
        'med_his_cardio_other', 'med_his_cardio_attack', 'med_his_cardio_aortic', 'med_his_cardio_artery', 
        'med_his_cardio_enlarge', 'med_his_cardio_regur', 'med_his_cardio_fail', 'med_his_cardio_breath', 
        'med_his_cardio_fluid', 'med_his_cardio_diuretics', 'med_his_arry', 'med_his_afib', 'med_his_aflutter', 
        'med_his_svt', 'med_his_vt', 'med_his_vfib', 'med_his_pacs', 'med_his_pvcs', 'med_his_arr_other', 
        'med_his_arr_other1', 'med_his_que_other___1', 'med_his_que_other___2', 'med_his_que_other___3', 
        'med_his_que_other___4', 'med_his_que_other___5', 'med_his_que_other___6', 'med_his_que_other___7', 
        'med_his_question_other1', 'med_his_arrhy_device', 'med_his_arrhy_neuro', 'med_his_allergy', 
        'med_his_childbearing', 'med_his_preg', 'medical_history_complete', 'self_reported_health_social_engagement_timestamp', 
        'montreal_cognitive_assessment_moca_timestamp', 'moca_1', 'moca_2', 'visuospatial', 'moca_4', 'moca_5', 'moca_6', 
        'naming', 'moca_7', 'moca_8', 'moca_9', 'moca_10', 'attention', 'moca_11', 'moca_12', 'language', 'moca_13', 
        'abstraction', 'moca_14', 'moca_15', 'orientation', 'bathroom_data_collection_timestamp', 'data_bathroom_id', 
        'data_start_time1', 'std_sys_bp1', 'std_dia_bp1', 'data_time_link', 'data_signal', 'data_start_time2', 
        'std_sys_bp2', 'std_dia_bp2', 'data_time_link_2', 'data_signal_2', 'data_start_time3', 'std_sys_bp3', 
        'std_dia_bp3', 'data_time_link_3', 'data_signal_3', 'bathroom_data_collection_complete', 
        'participant_experience_timestamp', 'initial_phone_eligibility_assessment_timestamp', 
        'physical_health_assessment_timestamp', 'final_eligibility_assessment_timestamp'
    ]

    # Delete specified columns
    df.drop(columns=columns_to_delete, inplace=True)

    # Rename columns
    df.rename(columns={'moca_3': 'moca_clock_draw', 'delayed_recall': 'moca_delayed_recall'}, inplace=True)

    # Clean and validate record_id
    df['record_id'] = df['record_id'].str.strip()
    pattern = re.compile(r'^CBP-\d{4}$')
    df = df[df['record_id'].apply(lambda x: bool(pattern.match(x)))]

    # Mapping of column names to their corresponding labels
    column_label_mapping = {
        'demo_gender': {0: 'Male', 1: 'Female'},
        'demo_race': {
            1: 'White or Caucasian', 2: 'Black or African-American', 3: 'Hispanic or Latino', 
            4: 'Asian or Pacific Islander', 5: 'Native American or Alaskan American', 
            6: 'Multiracial or Biracial', 7: 'Not listed here'
        },
        'demo_highest_edu': {
            1: '12th grade or less', 2: 'High school graduate', 3: 'Some college', 
            4: 'Associate\'s Degree', 5: 'Bachelor\'s Degree', 6: 'Master\'s Degree', 
            7: 'Doctorate or higher'
        },
        'demo_current_employ': {
            1: 'Retired', 2: 'Part-time employee', 3: 'Full-time employee', 4: 'Unemployed'
        },
        'demo_military_active': {1: 'Active duty', 2: 'Retired'},
        'demo_years_served': {1: 'Served < 5 years', 2: 'Served 5-10 years', 3: 'Served >10 years'},
        'demo_current_marital': {
            1: 'Single', 2: 'Married', 3: 'Domestic Partnership', 4: 'Widow', 
            5: 'Divorced', 6: 'Separated', 7: 'Other'
        },
        'demo_living_arrange': {
            1: 'Living by yourself', 2: 'Living with one person', 3: 'Living with more than one person', 
            4: 'Other'
        },
        'exc_age': {1: 'Yes', 0: 'No'},
        'exc_eng': {1: 'Yes', 0: 'No'},
        'exc_loc': {1: 'Yes', 0: 'No'},
        'exc_sit': {1: 'Yes', 0: 'No'},
        'exc_stand': {1: 'Yes', 0: 'No'},
        'exc_mob': {1: 'Yes', 0: 'No'},
        'exc_res': {1: 'Yes', 0: 'No'},
        'exc_consent': {1: 'Yes', 0: 'No'},
        'exc_mcs': {1: 'Yes', 0: 'No'},
        'exc_valve': {1: 'Yes', 0: 'No'},
        'exc_afib': {1: 'Yes', 0: 'No'},
        'exc_dialysis': {1: 'Yes', 0: 'No'},
        'exc_allergies': {1: 'Yes', 0: 'No'},
        'exc_skin': {1: 'Yes', 0: 'No'},
        'exc_hr': {1: 'Yes', 0: 'No'},
        'exc_spo2': {1: 'Yes', 0: 'No'},
        'exc_weight': {1: 'Yes', 0: 'No'},
        'exc_bmi': {1: 'Yes', 0: 'No'},
        'exc_eligible': {1: 'Eligible', 0: 'Ineligible'},
        'rating_social_eng': {
            1: 'Not very socially engaged', 2: 'Somewhat socially engaged', 3: 'Moderately socially engaged', 
            4: 'Quite socially engaged', 5: 'Very socially active'
        },
        'phy_skin': {
            1: 'Type I (Pale white)', 2: 'Type II (Fair)', 3: 'Type III (Medium)', 
            4: 'Type IV (Olive)', 5: 'Type V (Brown)', 
            6: 'Type VI (Very Dark)'
        },
        'exc_eligible_2': {1: 'Eligible', 0: 'Ineligible'},
        'pe_easy': {i: f"{i}" for i in range(1, 11)},
        'pe_valuable': {i: f"{i}" for i in range(1, 11)},
        'demographics_complete': {0: 'Incomplete', 1: 'Unverified', 2: 'Complete'},
        'self_reported_health_social_engagement_complete': {0: 'Incomplete', 1: 'Unverified', 2: 'Complete'},
        'montreal_cognitive_assessment_moca_complete': {0: 'Incomplete', 1: 'Unverified', 2: 'Complete'},
        'physical_health_assessment_complete': {0: 'Incomplete', 1: 'Unverified', 2: 'Complete'},
        'final_eligibility_assessment_complete': {0: 'Incomplete', 1: 'Unverified', 2: 'Complete'},
        'participant_experience_complete': {0: 'Incomplete', 1: 'Unverified', 2: 'Complete'}
    }

    # Convert numeric values to their corresponding labels based on the mapping
    for column, mapping in column_label_mapping.items():
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna(df[column])

    # First, create a mapping of record_id to exc_eligible from screening_arm_1
    record_eligibility_mapping = df[df['redcap_event_name'] == 'screening_arm_1'][['record_id', 'exc_eligible']]

    # Merge this mapping with the original DataFrame to propagate exc_eligible values to visit_1_arm_1 rows
    df = df.merge(record_eligibility_mapping, on='record_id', suffixes=('', '_from_screening'))

    # Now filter rows with 'exc_eligible' = 'Eligible' based on the propagated values
    df = df[df['exc_eligible_from_screening'] == 'Eligible']

    # Drop the temporary column used for merging
    df.drop(columns=['exc_eligible_from_screening'], inplace=True)

    # Divide into two dataframes
    df_screening = df[df['redcap_event_name'] == 'screening_arm_1'].drop(columns=['pe_easy', 'pe_valuable', 'participant_experience_complete']).copy()
    df_visit = df[df['redcap_event_name'] == 'visit_1_arm_1'][['record_id',  'exc_eligible', 'pe_easy', 'pe_valuable', 'participant_experience_complete']].copy()

    # Ensure exc_eligible is in both dataframes
    df_screening['exc_eligible'] = df_screening['exc_eligible']
    df_visit['exc_eligible'] = df_visit['exc_eligible']

    df_screening = df_screening.drop(columns=[
        'redcap_event_name', 'exc_age', 'exc_eng', 'exc_loc', 'exc_sit', 'exc_stand', 'exc_mob', 'exc_res', 
        'exc_consent', 'exc_mcs', 'exc_valve', 'exc_afib', 'exc_dialysis', 'exc_allergies', 'exc_skin', 
        'exc_hr', 'exc_spo2', 'exc_weight', 'exc_bmi', 'initial_phone_eligibility_assessment_complete'
    ])

    # Create a new column 'sub_moca_score'
    df_screening['sub_moca_score'] = df_screening[['moca_clock_draw', 'moca_delayed_recall']].sum(axis=1)

    # Define a function to assign color based on sub_moca_score
    def assign_moca_color(score):
        if score in [6, 7, 8]:
            return 'Green'
        elif score in [3, 4, 5]:
            return 'Yellow'
        elif score in [0, 1, 2]:
            return 'Red'
        else:
            return 'Unknown'

    # Apply the function to create the 'sub_moca_color' column
    df_screening['sub_moca_color'] = df_screening['sub_moca_score'].apply(assign_moca_color)

    # Define functions for plotting
    def add_value_labels(ax, data):
        """Add labels with count and percentage to the end of each bar in a bar chart."""
        total = len(data)
        for rect in ax.patches:
            height = rect.get_height()
            percentage = (height / total) * 100
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.1, 
                    f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')

    def adjust_yaxis(ax):
        """Adjust y-axis to make it more visually appealing."""
        if len(ax.patches) > 0:
            highest_count = max([p.get_height() for p in ax.patches])
            upper_limit = highest_count + (0.2 * highest_count)
            ax.set_ylim(0, upper_limit)
            step = max(1, int(highest_count / 5))
            ax.set_yticks(np.arange(0, upper_limit, step=step))

    # Streamlit sidebar options
    option = st.sidebar.selectbox(
        'Which visualization would you like to see?',
        ('Eligibility/Visit_1 Completion', 'Demographics', 'Overall Health', 'MoCA Score Distribution', 'Physical Health')
    )

    # Eligibility/Visit_1 Completion visualization
    if option == 'Eligibility/Visit_1 Completion':
        fig, axs = plt.subplots(2, 1, figsize=(7, 10), tight_layout=True)
        
        # Adjust the DataFrame for plotting purposes
        df_screening['eligibility_status'] = df_screening['exc_eligible']
        
        # Eligibility Status
        sns.countplot(x='eligibility_status', data=df_screening, palette='pastel', ax=axs[0])
        axs[0].set_title('Eligibility Status')
        axs[0].set_xlabel('')  # Remove x-axis label
        add_value_labels(axs[0], df_screening)
        adjust_yaxis(axs[0])

        # Visit 1 Completion Status
        df_visit['visit_1_complete'] = df_visit['participant_experience_complete'].apply(lambda x: 'Completed' if x == 'Complete' else 'Incomplete')

        sns.countplot(x='visit_1_complete', data=df_visit, palette='pastel', ax=axs[1])
        axs[1].set_title('Visit 1 Completion Status')
        axs[1].set_xlabel('')  # Remove x-axis label
        add_value_labels(axs[1], df_visit)
        adjust_yaxis(axs[1])


        st.pyplot(fig)

    elif option == 'Demographics':
        # Define the order for education levels
        education_order = [
            'High school graduate', '12th grade or less', 'Some college',
            'Associate\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'Doctorate or higher'
        ]
        race_order = [
            'White or Caucasian', 'Black or African-American', 
            'Asian or Pacific Islander', 'Hispanic or Latino',
            'Multiracial or Biracial', 'Not listed here', 'Native American or Alaskan American'
        ]

        # Convert the 'demo_highest_edu' and 'demo_race' columns to categorical types with the specified order
        df_screening['demo_highest_edu'] = pd.Categorical(
            df_screening['demo_highest_edu'], categories=education_order, ordered=True
        )
        df_screening['demo_race'] = pd.Categorical(
            df_screening['demo_race'], categories=race_order, ordered=True
        )

        demographic_cols = ['demo_gender', 'demo_race', 'demo_highest_edu', 'demo_current_employ', 'demo_current_marital']
        for col in demographic_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            if col == 'demo_highest_edu':
                valid_categories = df_screening[col].dropna().value_counts().index.tolist()
                valid_categories = [cat for cat in education_order if cat in valid_categories]
                df_filtered = df_screening[df_screening[col].isin(valid_categories)]
                plot = sns.countplot(x=col, data=df_filtered, palette='pastel', order=valid_categories, ax=ax)
            elif col == 'demo_race':
                valid_categories = df_screening[col].dropna().value_counts().index.tolist()
                valid_categories = [cat for cat in race_order if cat in valid_categories]
                df_filtered = df_screening[df_screening[col].isin(valid_categories)]
                plot = sns.countplot(x=col, data=df_filtered, palette='pastel', order=valid_categories, ax=ax)
            else:
                plot = sns.countplot(x=col, data=df_screening, palette='pastel', ax=ax)
            add_value_labels(ax, df_screening[col].dropna())  # Pass the filtered DataFrame directly
            adjust_yaxis(ax)  # Adjust y-axis based on non-null counts
            plt.title(f"{col.replace('demo_', '').replace('_', ' ').title()} Distribution")
            plt.xlabel('')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Special additional graph for gender distribution within each race category
            if col == 'demo_race':
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x='demo_race', hue='demo_gender', data=df_filtered, palette='pastel', order=valid_categories, ax=ax)
                plt.title('Gender Distribution Within Each Race Category')
                plt.xlabel('Race')
                plt.ylabel('Count')
                plt.legend(title='Gender')
                plt.xticks(rotation=45)
                add_value_labels(ax, df_screening[['demo_race', 'demo_gender']].dropna())  # Adjusted for hue-grouped data
                adjust_yaxis(ax)
                st.pyplot(fig)

        # Calculate mean and standard deviation
        mean_age = df_screening['demo_age'].mean()
        std_dev_age = df_screening['demo_age'].std()

        # Display mean and standard deviation
        st.write(f"**Mean Age:** {mean_age:.2f}")
        st.write(f"**Standard Deviation:** {std_dev_age:.2f}")

        # Plot histogram for demo_age
        fig, ax = plt.subplots(figsize=(10, 6))
        age_bins = range(50, 91, 5)
        sns.histplot(df_screening['demo_age'].dropna(), bins=age_bins, kde=False, color='skyblue', ax=ax)
        add_value_labels(ax, df_screening['demo_age'].dropna())  # Adjusted for histogram
        adjust_yaxis(ax)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.xticks(age_bins)
        st.pyplot(fig)



    # Overall Health visualization
    elif option == 'Overall Health':
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        
        # Overall Health Rating Distribution
        sns.boxplot(y='rating_overall_health', data=df_screening, color='#FFA07A', orient='v', showmeans=True, ax=ax)
        ax.set_title('Overall Health Rating Distribution')
        
        # Calculating mean, median, and standard deviation for overall health ratings
        mean_rating = df_screening['rating_overall_health'].mean()
        median_rating = df_screening['rating_overall_health'].median()
        std_rating = df_screening['rating_overall_health'].std()
        
        # Position stats in the rightmost corner of the plot
        stats_text = f'Mean: {mean_rating:.2f}\nMedian: {median_rating}\nStd: {std_rating:.2f}'
        ax.text(1.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top')
        
        st.pyplot(fig)

    # MoCA Score Distribution visualization
    elif option == 'MoCA Score Distribution':

        education_order = [
            'High school graduate', '12th grade or less', 'Some college',
            'Associate\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'Doctorate or higher'
        ]

        # Shorter labels for x-axis
        education_labels_short = [
            'HS', '12th', 'College',
            'Asso', 'Bach', 'Mast', 'Doc'
        ]

        # Create a dictionary for mapping full names to shorter names
        education_label_map = dict(zip(education_order, education_labels_short))

        # Reorder the data
        df_screening['demo_highest_edu'] = pd.Categorical(
            df_screening['demo_highest_edu'], categories=education_order, ordered=True
        )

        # MoCA Score Distribution visualization
        if option == 'MoCA Score Distribution':
            # Create subplots for box plots and pie charts
            fig, axs = plt.subplots(4, 1, figsize=(8, 24), tight_layout=True)
            
            # Data and colors for each plot
            plot_data = {
                'Total MoCA Score Distribution': ('total_moca_2', '#20B2AA'),
                'Sub MoCA Score Distribution': ('sub_moca_score', '#FF6347'),
                'Sub MoCA Score Distribution by Gender': ('sub_moca_score', 'pastel', 'demo_gender'),
                'Sub MoCA Score Distribution by Highest Education Level': ('sub_moca_score', 'pastel', 'demo_highest_edu')
            }
            
            # Y-axis labels for each plot
            y_labels = {
                'total_moca_2': 'Total MoCA Score',
                'sub_moca_score': 'Sub MoCA Score'
            }
            
            # X-axis labels for plots with grouped data
            x_labels = {
                'demo_gender': 'Gender',
                'demo_highest_edu': 'Education Level'
            }
            
            # Iterate over each plot
            for i, (title, info) in enumerate(plot_data.items()):
                if len(info) == 2:
                    data_col, color = info
                    sns.boxplot(y=data_col, data=df_screening, color=color, orient='v', showmeans=True, ax=axs[i])
                else:
                    data_col, palette, hue_col = info
                    sns.boxplot(x=hue_col, y=data_col, data=df_screening, ax=axs[i], palette=palette,
                                order=education_order if hue_col == 'demo_highest_edu' else None)
                
                axs[i].set_title(title)
                
                # Set y-axis label
                if data_col in y_labels:
                    axs[i].set_ylabel(y_labels[data_col])
                
                # Set x-axis label
                if len(info) == 3:
                    axs[i].set_xlabel(x_labels[hue_col])
                
                # Calculate mean, median, std
                if i == 2 or i == 3:
                    # If grouped by another column, calculate these stats per group
                    group_stats = df_screening.groupby(hue_col)[data_col].agg(['mean', 'median', 'std']).round(2)
                    text_info = ""
                    for name, group in group_stats.iterrows():
                        text_info += f"{name} - M: {group['mean']}, Md: {group['median']}, Std: {group['std']}\n"
                    axs[i].text(1.05, 0.95, text_info, transform=axs[i].transAxes, ha='left', va='top', fontsize=9)
                    # For x-axis labels in education level plot, replace with shorter names
                    if hue_col == 'demo_highest_edu':
                        axs[i].set_xticklabels([education_label_map[label.get_text()] for label in axs[i].get_xticklabels()])
                else:
                    mean = df_screening[data_col].mean()
                    median = df_screening[data_col].median()
                    std = df_screening[data_col].std()
                    # Position stats in the rightmost corner
                    axs[i].text(1.05, 0.95, f'Mean: {mean:.2f}\nMedian: {median}\nStd: {std:.2f}', 
                                transform=axs[i].transAxes, ha='left', va='top', fontsize=9)

            st.pyplot(fig)

        # Define color schemes for the charts
        sub_moca_colors = {'Green': '#209c05', 'Yellow': '#f2ce02', 'Red': '#ff0a0a'}

        # Overall Sub MoCA Color Distribution
        st.subheader('Overall Sub MoCA Color Distribution')
        color_counts = df_screening['sub_moca_color'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(color_counts, labels=[f'{i}\n({v})' for i, v in color_counts.items()], colors=[sub_moca_colors[x] for x in color_counts.index], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Gender-specific Sub MoCA Color Distribution
        col1, col2 = st.columns(2)
        for col, gender in zip([col1, col2], ['Male', 'Female']):
            with col:
                st.subheader(f'{gender} Sub MoCA Color Distribution')
                gender_color_counts = df_screening[df_screening['demo_gender'] == gender]['sub_moca_color'].value_counts()
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(gender_color_counts, labels=[f'{i}\n({v})' for i, v in gender_color_counts.items()], colors=[sub_moca_colors[x] for x in gender_color_counts.index], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

        # Education-level Sub MoCA Color Distribution
        st.subheader('Education-Level Sub MoCA Color Distribution')
        edu_levels = df_screening['demo_highest_edu'].unique()

        # Filter out empty education levels
        edu_levels = [edu for edu in edu_levels if not pd.isnull(edu)]

        # Iterate over education levels and create subplots
        for i, edu in enumerate(edu_levels):
            if i % 2 == 0:  # Start a new row for every two plots
                col1, col2 = st.columns(2)
            with col1 if i % 2 == 0 else col2:
                st.subheader(f'{edu}')
                edu_color_counts = df_screening[df_screening['demo_highest_edu'] == edu]['sub_moca_color'].value_counts()
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(edu_color_counts, labels=[f'{i}\n({v})' for i, v in edu_color_counts.items()], colors=[sub_moca_colors[x] for x in edu_color_counts.index], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

    # Physical Health visualization
    elif option == 'Physical Health':
        st.header('Statistical Description of Physical Measurements')
        physical_measures_cols = ['phy_height_inch', 'phy_weight_lb', 'phy_bmi', 'phy_arm', 'phy_sternal']
        physical_measures_names = ['Height (inches)', 'Weight (lbs)', 'BMI', 'Arm circumference (cm)', 'Sternal length (inch)']

        # Calculating descriptive statistics and transposing the result
        physical_desc = df_screening[physical_measures_cols].describe().transpose()

        # Rename the index to have more meaningful names
        physical_desc.index = physical_measures_names

        # Use string formatting to ensure numbers are rounded and displayed with two decimal places
        formatted_physical_desc = physical_desc.applymap(lambda x: f'{x:.2f}')

        # Displaying the table
        st.table(formatted_physical_desc)

        # Adding histogram for phy_skin
        st.subheader('Skin Type Distribution')
        skin_types = ['Type I (Pale white)', 'Type II (Fair)', 'Type III (Medium)', 'Type IV (Olive)', 'Type V (Brown)', 'Type VI (Very Dark)']
        skin_counts = df_screening['phy_skin'].value_counts().reindex(skin_types, fill_value=0)
        colors = ['#FFDDC1', '#FCCB95', '#F8B888', '#D8A378', '#C68642', '#8C6239']
        
        # Explicitly create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=skin_types, y=skin_counts.values, palette=colors, ax=ax)

        # Add labels with count and percentage to the end of each bar in a bar chart
        total = len(df_screening['phy_skin'])
        for rect in ax.patches:
            height = rect.get_height()
            percentage = (height / total) * 100
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.1, 
                    f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')

        ax.set_xticklabels(skin_types, rotation=45)
        ax.set_ylabel('Number of Participants')
        ax.set_title('Distribution of Skin Types')

        # Adjust y-axis to make it more visually appealing
        if len(ax.patches) > 0:
            highest_count = max([p.get_height() for p in ax.patches])
            upper_limit = highest_count + (0.2 * highest_count)
            ax.set_ylim(0, upper_limit)
            ax.set_yticks(np.arange(0, upper_limit, step=max(1, highest_count // 5)))

        # Use the figure object with st.pyplot()
        st.pyplot(fig)
