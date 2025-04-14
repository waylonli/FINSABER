from datetime import datetime, timedelta
import requests
import os

def get_start_and_end_timestamps(input_datetime):
    # Extract the date part from the input datetime
    date_part = input_datetime.date()

    # Get the starting timestamp (midnight of the given date)
    start_datetime = datetime.combine(date_part, datetime.min.time())
    start_timestamp = int(start_datetime.timestamp())

    # Get the end timestamp (11:59:59 PM of the given date)
    end_datetime = datetime.combine(date_part, datetime.max.time())
    end_timestamp = int(end_datetime.timestamp())

    return start_timestamp, end_timestamp

class RapidAPIs:
    def __init__(self):
        try:
            self.X_RapidAPI_Key = os.environ.get("RAPIDAPI_KEY")
        except:
            print("Please config your RAPIDAPI_KEY token. The token could be found at https://rapidapi.com/apidojo/api/seeking-alpha")

    def get_seekingAlpha_analysis(self, stock, start_timestamp, end_timestamp):

        # the free package is 500 times/mon. And the api is date-based

        summaries = []
        sentiments = []
        timestamps = []
        expert_knowledge_ids = []
        urls = []
        images = []
        titles = []
        authors = []

        headers = {
            "X-RapidAPI-Key": self.X_RapidAPI_Key, 
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }

        # start_timestamp, end_timestamp = get_start_and_end_timestamps(today)

        url = "https://seeking-alpha.p.rapidapi.com/analysis/v2/list"
        querystring = {"id": stock, "size":"40", "number":"1", "since": str(int(start_timestamp)), "until": str(int(end_timestamp))}

        try:
            response = requests.get(url, headers=headers, params=querystring).json()
            print(response)
            analysis_list = response['data']
        except Exception as e:
            print(f"Error in seekingAlphaAPI {e}")
            analysis_list = []

        for analysis in analysis_list:
            try:
                url = "https://seeking-alpha.p.rapidapi.com/analysis/v2/get-details"
                querystring = {"id":analysis['id']}
                response = requests.get(url, headers=headers, params=querystring).json()
                title = response['data']['attributes']['title']
                # concate summary into a single string
                summary = ""
                summary_id = 0
                for paragraph in response['data']['attributes']['summary']:
                    # number the paragraphs
                    paragraph = str(summary_id) + ". " + paragraph+" "
                    summary += paragraph
                    summary_id += 1

                sentiment= response['included'][-1]['attributes']['type']
                timestamp= response['data']['attributes']['publishOn']
                expert_knowledge_id= response['data']['id']
                url= response['data']['links']['canonical']
                image= response['data']['links']['uriImage']
                author=response['included'][0]["attributes"]["slug"]
                parsing_status = "success"

            except Exception as e:
                print(f"Error in seekingAlphaAPI {e}")
                parsing_status = "failed"
            if parsing_status=="success":
                summaries.append(summary)
                sentiments.append(sentiment)
                timestamps.append(timestamp)
                expert_knowledge_ids.append(expert_knowledge_id)
                urls.append(url)
                images.append(image)
                titles.append(title)
                authors.append(author)

        return {
            "title": titles,
            "summary": summaries,
            "sentiment": sentiments,
            "timestamp": timestamps,
            "expert_knowledge_id": expert_knowledge_ids,
            "url": urls,
            "image": images,
            "author": authors
        }
    
    def get_stockSentiment_sentiment(self, stock):
        # the free package is 100 times/mon. And this api is realtime only

        summaries = []
        sentiments = []

        headers = {
            "X-RapidAPI-Key": self.X_RapidAPI_Key,
            "X-RapidAPI-Host": "stock-sentiment-api.p.rapidapi.com"
        }

        url = "https://stock-sentiment-api.p.rapidapi.com/stock_news_sentiment/"

        querystring = {"ticker":stock}

        response = requests.get(url, headers=headers, params=querystring).json()
        count = len(response)
        #TODO: Find a way to project the score to a text description
        for data in response:
            summaries.append(data['title'])

            compound = data['compound']
            if compound > 0.5:
                sentiment = 'very_bullish'
            elif compound > 0.25:
                sentiment = 'bullish'
            elif compound < -0.5:
                sentiment = 'very_bearish'
            elif compound < -0.25:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            sentiments.append(sentiment)
        
        return {
            'summaries': summaries,
            'sentiments': sentiments,
            'sentiments_score_compound': compound
        }


if __name__ == "__main__":
    date_time = datetime(2022, 1, 28, 21, 20)
    apis = RapidAPIs()

    # print(apis.seekingAlphaAPI("aapl", date_time))
    # print(apis.stockSentimentAPI("aapl"))