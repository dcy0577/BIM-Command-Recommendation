### only used in the small dataset!###

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw_bar_plot(data, title, xlabel, ylabel, rotation=45, ha='right', save_path=None):
    plt.figure(figsize=(15, 14))
    ax = data.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha=ha)
    # add labels to each bar in the plot
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

def draw_bar_plot_sns(data, title, xlabel, ylabel, save_path=None):
    # Plotting the top 50 menu actions based on their counts
    plt.figure(figsize=(15, 12))
    sns.barplot(y=data.index, x=data.values, palette="magma")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

def draw_pie_plot(data, title, save_path=None):
    plt.figure(figsize=(12, 12))
    data.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title(title)
    plt.ylabel('') 
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

def save_stats_to_file(data_frame, save_path):
    data_frame.to_csv(save_path, index=False)


def combine_csv_to_pkl(directory_path = '/mnt/c/Users/ge25yak/Downloads/VW_User_Logs_for_TUM_2023-04-12_1000SNs_anonymized/VW_User_Logs_for_TUM_2023-04-12_1000SNs_anonymized'
):
    if not os.path.exists(os.path.join('data', 'combined.pkl')):
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        # Load all CSV files into a list of dataframes
        list_of_dataframes = []
        for file in csv_files:
            full_path = os.path.join(directory_path, file)
            df = pd.read_csv(full_path)
            list_of_dataframes.append(df)
        # Concatenate all dataframes into a single dataframe
        combined_df = pd.concat(list_of_dataframes)
        # save the combined dataframe to a pkl file
        combined_df.to_pickle(os.path.join('data', 'combined.pkl'))
    else:
        print('combined.pkl already exists')

def plot_cat_distribution():
    if os.path.exists(os.path.join('data', 'combined.pkl')):

        # load the combined dataframe from a pkl file
        combined_df = pd.read_pickle(os.path.join('data', 'combined.pkl'))

        # counts the number of rows with each unique value of the 'cat' column
        category_counts = combined_df['cat'].value_counts()

        # draws a bar plot of the category counts
        draw_bar_plot(category_counts, 'Category Content Counts', 'Category Content', 'Count', rotation=45, ha='right', save_path='category_content_counts_bar.png')
        # draws a pie chart of the category counts
        draw_pie_plot(category_counts, 'Category Content Counts', save_path='category_content_counts_pie.png')

def plot_commands_distribution():
    if os.path.exists(os.path.join('data', 'combined.pkl')):
        # load the combined dataframe from a pkl file
        combined_df = pd.read_pickle(os.path.join('data', 'combined.pkl'))

        # # Remove rows with messages starting with the specified prefixes
        # prefixes_to_remove = PREFIXES_TO_REMOVE
        # for prefix in prefixes_to_remove:
        #     combined_df = combined_df[~combined_df['message'].str.startswith(prefix)]

        # Remove any text matching the pattern (MAX-<any number>)
        combined_df['message'] = combined_df['message'].str.replace(r'\(MAX-\d+\)', '', regex=True)

        # counts the number of rows with each unique value of the 'message' column
        # plot only top 50 commands
        command_counts = combined_df['message'].value_counts().head(50)

        # draws a bar plot of the command counts
        fig, ax = plt.subplots(figsize=(15, 14))
        command_counts.plot(kind='bar', color='skyblue', ax=ax)
        plt.title('Top-50 Commands Content Counts without any filtering')
        plt.xlabel('Commands Content')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        # add labels to each bar in the plot
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.000e}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.tight_layout()
        plt.savefig('plots/Commands_content_counts_bar.png')

        # draws a pie chart of the command counts
        draw_pie_plot(command_counts, 'Top-50 Commands Content Counts without any filtering', save_path='plots/Commands_content_counts_pie.png')


def plot_commands_distribution_in_different_cats():
    if os.path.exists(os.path.join('data', 'combined.pkl')):
        # load the combined dataframe from a pkl file
        combined_df = pd.read_pickle(os.path.join('data', 'combined.pkl'))

        # get rows that belong to 'Menu' in the 'cat' column
        menu_df = combined_df[combined_df['cat'] == 'Menu']
        # save the stats to a csv file
        stats = menu_df['message'].value_counts().reset_index()
        stats.columns = ['message content', 'count']
        save_stats_to_file(stats, 'plots/menu.csv')
        # counts the top-50 number of rows with each unique value of the 'message' column
        menu_command_counts = menu_df['message'].value_counts().head(50)
        # draws a bar plot of the command counts
        # draw_bar_plot(menu_command_counts, 'Top-50 Menu Commands Content Counts without any filtering', 'Commands Content', 'Count', rotation=45, ha='right', save_path='menu_commands_content_counts_bar.png')
        draw_bar_plot_sns(menu_command_counts, 'Top-50 Menu Commands Content Counts without any filtering', 'Commands Content', 'Count', save_path='plots/menu_commands_content_counts_bar_sns.png')
        

        # get rows that belong to 'Tool' in the 'cat' column
        tool_df = combined_df[combined_df['cat'] == 'Tool']
        # save the stats to a csv file
        stats = tool_df['message'].value_counts().reset_index()
        stats.columns = ['message content', 'count']
        save_stats_to_file(stats, 'plots/tool.csv')
        # counts the top-50 number of rows with each unique value of the 'message' column
        tool_command_counts = tool_df['message'].value_counts().head(50)
        # draws a bar plot of the command counts
        # draw_bar_plot(tool_command_counts, 'Top-50 Tool Commands Content Counts without any filtering', 'Commands Content', 'Count', rotation=45, ha='right', save_path='tool_commands_content_counts_bar.png')
        draw_bar_plot_sns(tool_command_counts, 'Top-50 Tool Commands Content Counts without any filtering', 'Commands Content', 'Count', save_path='plots/tool_commands_content_counts_bar_sns.png')


        # get rows that belong to 'UNDO' in the 'cat' column
        undo_df = combined_df[combined_df['cat'] == 'UNDO']
        # save the stats to a csv file
        stats = undo_df['message'].value_counts().reset_index()
        stats.columns = ['message content', 'count']
        save_stats_to_file(stats, 'plots/undo.csv')
        # counts the top-50 number of rows with each unique value of the 'message' column
        undo_command_counts = undo_df['message'].value_counts().head(50)
        # draws a bar plot of the command counts
        # draw_bar_plot(undo_command_counts, 'Top-50 UNDO Commands Content Counts without any filtering', 'Commands Content', 'Count', rotation=45, ha='right', save_path='undo_commands_content_counts_bar.png')
        draw_bar_plot_sns(undo_command_counts, 'Top-50 UNDO Commands Content Counts without any filtering', 'Commands Content', 'Count', save_path='plots/undo_commands_content_counts_bar_sns.png')


def plot_time_disturbution():
    if os.path.exists(os.path.join('data', 'combined.pkl')):
        # load the combined dataframe from a pkl file
        combined_df = pd.read_pickle(os.path.join('data', 'combined.pkl'))
        # Convert the 'ts' column to datetime format with milliseconds
        combined_df['ts'] = pd.to_datetime(combined_df['ts'], format='%Y-%m-%d %H:%M:%S.%f')
        # Extracting date and hour to create a new time label
        combined_df['date_hour'] = combined_df['ts'].dt.strftime('%Y-%m-%d %H')

        # Grouping by the new time label and counting the number of logs for each hour
        hourly_logs_multi_day = combined_df.groupby('date_hour').size()

        # Plotting the user activity curve for multiple days using a line plot
        plt.figure(figsize=(15, 7))
        sns.lineplot(x=hourly_logs_multi_day.index, y=hourly_logs_multi_day.values, marker='o', color="blue")
        plt.title("user activity curve")
        plt.xlabel("date and hour")
        plt.ylabel("amount of records")
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('plots/user_activity_curve.png')


def session_stats_before_after_filtering(output_path_bef='plots/session_stats.csv', output_path_aft='plots/session_stats_after_filtering.csv'):
    # before filtering
    if os.path.exists(os.path.join('data', 'combined.pkl')):
        # load the combined dataframe from a pkl file
        combined_df = pd.read_pickle(os.path.join('data', 'combined.pkl'))
        # Convert the 'ts' column to datetime format
        combined_df['ts'] = pd.to_datetime(combined_df['ts'])
        # Group by the 'session_anonymized' column and calculate the session length for each session
        session_lengths = combined_df.groupby('session_anonymized').agg({
            'ts': ['min', 'max']
        })
        session_lengths['session_length'] = session_lengths[('ts', 'max')] - session_lengths[('ts', 'min')]
        # Calculate the total number of rows for each session
        session_lengths['total_rows'] = combined_df.groupby('session_anonymized').size()

        # Reset column levels for easier indexing
        session_lengths.columns = session_lengths.columns.droplevel(1)
        session_lengths[['session_length', 'total_rows']].to_csv(output_path_bef)

    # after filtering
    if os.path.exists(os.path.join('data', 'filtered.pkl')):
        filtered_df = pd.read_pickle(os.path.join('data', 'filtered.pkl'))
        # Convert the UNIX timestamp to datetime format
        filtered_df['ts_unix'] = pd.to_datetime(filtered_df['ts'], unit='s')

        # Group by the 'session_anonymized' column and calculate the session length for each session
        session_lengths_unix = filtered_df.groupby('session_anonymized').agg({
        'ts_unix': ['min', 'max']
        })
        session_lengths_unix['session_length'] = session_lengths_unix[('ts_unix', 'max')] - session_lengths_unix[('ts_unix', 'min')]

        # Calculate the total number of rows for each session
        session_lengths_unix['total_rows'] = filtered_df.groupby('session_anonymized').size()

        # Reset column levels for easier indexing
        session_lengths_unix.columns = session_lengths_unix.columns.droplevel(1)
        session_lengths_unix[['session_length', 'total_rows']].to_csv(output_path_aft)

def plot_session_stats(data_path = 'plots/session_stats.csv'):
    if os.path.exists(data_path):
        session_data = pd.read_csv(data_path)
        # Calculate the mean values
        mean_total_rows = session_data['total_rows'].mean()
        mean_session_length_minutes = session_data['session_length_minutes'].mean()

        # Create a figure with two subplots: one for each distribution
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        # Plot the distribution of total rows of sessions with more x-axis labels and mean line
        sns.histplot(session_data['total_rows'], bins=100, kde=True, ax=ax[0])
        ax[0].set_title('Distribution of Total Rows of Sessions')
        ax[0].set_xlabel('Total Rows')
        ax[0].set_ylabel('Frequency')
        ax[0].set_xticks(range(0, session_data['total_rows'].max(), int(session_data['total_rows'].max() / 10)))
        ax[0].axvline(mean_total_rows, color='r', linestyle='--')
        ax[0].text(mean_total_rows, ax[0].get_ylim()[1]*0.9, f'Mean: {mean_total_rows:.2f}', color='r', rotation=90, verticalalignment='center')

        # Plot the distribution of session length (in minutes) with more x-axis labels and mean line
        sns.histplot(session_data['session_length_minutes'], bins=100, kde=True, ax=ax[1])
        ax[1].set_title('Distribution of Session Length')
        ax[1].set_xlabel('Session Length (minutes)')
        ax[1].set_ylabel('Frequency')
        ax[1].set_xticks(range(0, int(session_data['session_length_minutes'].max()), 
                        int(session_data['session_length_minutes'].max() / 10)))
        ax[1].axvline(mean_session_length_minutes, color='r', linestyle='--')
        ax[1].text(mean_session_length_minutes, ax[1].get_ylim()[1]*0.9, f'Mean: {mean_session_length_minutes:.2f}', color='r', rotation=90, verticalalignment='center')

        # Display the plot with a tight layout
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    ### only used in the small dataset!###

    # combine_csv_to_pkl()
    # plot_cat_distribution()
    plot_commands_distribution()
    plot_commands_distribution_in_different_cats()
    plot_time_disturbution()
    session_stats_before_after_filtering()
