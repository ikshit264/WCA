from urlextract import URLExtract
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string
import emoji

extract = URLExtract()

# Define custom stopwords by reading the file
with open("stop_hinglish.txt", "r") as file:
    hinglish_stopwords = set(
        file.read().splitlines()
    )  # Read the file line by line into a set

# Combine custom stopwords (hinglish + custom) with the English stopwords
custom_stopwords = {
    "<media",
    "omitted>",
    "media",
    "omitted",
    "{",
    "}",
    "-",
    "<",
    ">",
    "[",
    "]",
    "_",
    "},",
    "~",
}
STOP_WORDS = (
    STOPWORDS.union(hinglish_stopwords)
    .union(custom_stopwords)
    .union(string.punctuation)
)


def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    # 1. fetch the number of messages
    num_messages = df.shape[0]

    # 2. fetch the total number of words
    total_words = []
    for message in df["message"]:
        total_words.extend(message.split())

    # 3. fetch number of media messages
    total_media = df[df["message"] == "<Media omitted>\n"].shape[1]

    # 4. fetch number of links shared
    total_links = []
    for message in df["message"]:
        total_links.extend(extract.find_urls(message))

    return num_messages, len(total_words), total_media, len(total_links)


def most_busy_users(df):
    x = df["user"].value_counts().head()
    df = (
        round((df["user"].value_counts() / df.shape[0]) * 100, 2)
        .reset_index()
        .rename(columns={"index": "name", "user": "percent"})
    )
    return x, df


def create_wordcloud(selected_user, df):
    df = df[df["user"] != "group_notification"]
    df = df[df["message"] != "<Media omitted>\n"]
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    wc = WordCloud(
        width=800,
        height=800,
        min_font_size=10,
        background_color="black",
        stopwords=STOP_WORDS,
    )

    # Generate word cloud using message column
    df_wc = wc.generate(df["message"].str.cat(sep=" "))

    # Plot the word cloud using matplotlib and resize it for display
    plt.figure(
        figsize=(8, 4)
    )  # Downscale display to smaller size (e.g., 800x400 pixels)
    plt.imshow(df_wc, interpolation="bilinear")
    # plt.axis('off')  # Hide axis
    plt.show()

    return df_wc


def most_common_words(selectedUser, df):
    if selectedUser != "Overall":
        df = df[df["user"] == selectedUser]

    df = df[df["user"] != "group_notification"]
    df = df[df["message"] != "<Media omitted>\n"]

    words = []

    for message in df["message"]:
        for word in message.lower().split():
            if word not in STOP_WORDS:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))

    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    emojis = []
    for message in df["message"]:
        emojis.extend(
            [c for c in message if emoji.is_emoji(c)]
        )  # Use is_emoji() to check

    # Create a DataFrame with the count of each emoji
    emoji_df = pd.DataFrame(
        Counter(emojis).most_common(len(Counter(emojis))), columns=["emoji", "count"]
    )

    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = (
        df.groupby(["year", "month_num", "month"]).count()["message"].reset_index()
    )

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline["month"][i] + "-" + str(timeline["year"][i]))

    timeline["time"] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    daily_timeline = df.groupby("only_date").count()["message"].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["day_name"].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["month"].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    user_heatmap = df.pivot_table(
        index="day_name", columns="period", values="message", aggfunc="count"
    ).fillna(0)

    return user_heatmap
