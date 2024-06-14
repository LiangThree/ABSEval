import requests
from bs4 import BeautifulSoup


def get_google_search_results(query):
    query = query.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}&hl=en"
    print(f"url:{url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    response = requests.get(url, headers=headers)
    # soup = BeautifulSoup(response.text, 'html.parser')

    # Find the element containing the number of search results
    # result_stats = soup.find("div", {"id": "result-stats"})
    print(response.text)

    #
    # if result_stats:
    #     result_text = result_stats.text
    #     result_number = result_text.split("About ")[1].split(" results")[0]
    #     return result_number
    # else:
    #     return "No results found"


if __name__ == "__main__":
    query = "apple is a fruit"
    result_number = get_google_search_results(query)
    print("Number of search results:", result_number)
