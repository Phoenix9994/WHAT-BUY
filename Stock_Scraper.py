import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json

class TradingViewScraper:
    def __init__(self, headless=True):
        """
        Initialize the TradingView scraper
        
        Args:
            headless (bool): Whether to run browser in headless mode
        """
        self.base_url = "https://www.tradingview.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.headless = headless
        
    def setup_driver(self):
        """Setup Chrome WebDriver for dynamic content scraping"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    
    def click_load_more(self, driver, max_clicks=10):
        """
        Click the 'Load More' button multiple times to get more data
        
        Args:
            driver: Selenium WebDriver instance
            max_clicks (int): Maximum number of times to click load more
            
        Returns:
            int: Number of successful clicks
        """
        clicks = 0
        wait = WebDriverWait(driver, 10)
        
        print(f"Attempting to click 'Load More' up to {max_clicks} times...")
        
        while clicks < max_clicks:
            try:
                # Wait a bit before trying to find the button
                time.sleep(2)
                
                # Multiple possible selectors for the load more button
                load_more_selectors = [
                    "button[data-name='load-more-button']",
                    "button[data-name='load-more']",
                    ".tv-load-more-button",
                    "button.tv-button--secondary",
                    "button.tv-button--ghost",
                    ".tv-screener-toolbar__button--load-more",
                    "button:contains('Load more')",
                    "button:contains('Show more')"
                ]
                
                load_more_button = None
                
                # Try each selector
                for selector in load_more_selectors:
                    try:
                        if ':contains(' in selector:
                            # Handle text-based selection
                            buttons = driver.find_elements(By.TAG_NAME, "button")
                            for button in buttons:
                                button_text = button.text.lower()
                                if any(text in button_text for text in ['load more', 'show more', 'load', 'more']):
                                    if button.is_displayed() and button.is_enabled():
                                        load_more_button = button
                                        break
                        else:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            for element in elements:
                                if element.is_displayed() and element.is_enabled():
                                    load_more_button = element
                                    break
                        
                        if load_more_button:
                            break
                            
                    except NoSuchElementException:
                        continue
                
                if load_more_button:
                    # Scroll to the button to ensure it's visible
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_button)
                    time.sleep(1)
                    
                    # Get current row count before clicking
                    current_rows = len(driver.find_elements(By.CSS_SELECTOR, "tr[data-rowkey], tbody tr"))
                    
                    # Click the button using JavaScript to avoid interception
                    driver.execute_script("arguments[0].click();", load_more_button)
                    print(f"Clicked 'Load More' button #{clicks + 1}")
                    
                    # Wait for new content to load
                    time.sleep(4)
                    
                    # Check if new rows were added
                    new_rows = len(driver.find_elements(By.CSS_SELECTOR, "tr[data-rowkey], tbody tr"))
                    
                    if new_rows > current_rows:
                        print(f"New rows loaded: {new_rows - current_rows}")
                        clicks += 1
                    else:
                        print("No new rows loaded, assuming no more data available")
                        break
                        
                else:
                    print("Load more button not found or not clickable")
                    break
                    
            except Exception as e:
                print(f"Error clicking load more button: {e}")
                break
                
        print(f"Successfully clicked 'Load More' {clicks} times")
        return clicks
    
    def scrape_with_selenium(self, url, max_load_clicks=5, max_retries=3):
        """
        Scrape data using Selenium for dynamic content with load more functionality
        
        Args:
            url (str): URL to scrape
            max_load_clicks (int): Maximum number of times to click load more
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            list: List of dictionaries containing stock data
        """
        for attempt in range(max_retries):
            driver = None
            try:
                driver = self.setup_driver()
                print(f"Attempting to scrape {url} (Attempt {attempt + 1}/{max_retries})")
                
                driver.get(url)
                
                # Wait for the table to load
                wait = WebDriverWait(driver, 20)
                table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table, .tv-screener-table, [data-name='screener-table']")))
                
                # Additional wait for dynamic content to load
                print("Waiting for initial content to load...")
                time.sleep(5)
                
                # Click load more button multiple times
                if max_load_clicks > 0:
                    clicks = self.click_load_more(driver, max_load_clicks)
                    print(f"Load more process completed with {clicks} clicks")
                
                # Now extract all the data
                print("Extracting stock data...")
                
                # Try to find the table with stock data
                rows = driver.find_elements(By.CSS_SELECTOR, "tr[data-rowkey]")
                
                if not rows:
                    # Alternative selectors for TradingView's table structure
                    alternative_selectors = [
                        "tbody tr",
                        ".tv-screener-table__result-row",
                        ".tv-data-table tbody tr",
                        "table tr[role='row']"
                    ]
                    
                    for selector in alternative_selectors:
                        rows = driver.find_elements(By.CSS_SELECTOR, selector)
                        if rows:
                            print(f"Found rows using selector: {selector}")
                            break
                
                print(f"Found {len(rows)} rows to process")
                
                stocks_data = []
                
                for i, row in enumerate(rows):
                    try:
                        # Extract data from each row
                        cells = row.find_elements(By.TAG_NAME, "td")
                        
                        if len(cells) >= 2:  # Minimum columns needed
                            # Extract symbol and company name
                            symbol = ""
                            company_name = ""
                            
                            # Try different methods to extract symbol
                            try:
                                # Method 1: Look for symbol in first cell
                                symbol_element = cells[0].find_element(By.TAG_NAME, "a")
                                symbol = symbol_element.text.strip()
                                
                                # Try to get company name from title or data attributes
                                company_name = symbol_element.get_attribute("title") or ""
                                
                                # Alternative: look for description field
                                if not company_name:
                                    try:
                                        desc_element = row.find_element(By.CSS_SELECTOR, "[data-field='description'], [data-field-key='description']")
                                        company_name = desc_element.text.strip()
                                    except:
                                        pass
                                        
                            except:
                                # Fallback: just get text from first cell
                                symbol = cells[0].text.strip().split('\n')[0]
                            
                            if not symbol:
                                continue
                            
                            # Extract price (usually in second column)
                            price = ""
                            try:
                                price_element = cells[1]
                                price = price_element.text.strip()
                            except:
                                pass
                            
                            # Extract other columns
                            change = cells[2].text.strip() if len(cells) > 2 else ""
                            change_percent = cells[3].text.strip() if len(cells) > 3 else ""
                            volume = cells[4].text.strip() if len(cells) > 4 else ""
                            market_cap = cells[5].text.strip() if len(cells) > 5 else ""
                            
                            # Look for dividend yield in later columns
                            dividend_yield = ""
                            try:
                                for j in range(6, min(len(cells), 15)):
                                    cell_text = cells[j].text.strip()
                                    if '%' in cell_text and cell_text != '—' and cell_text != '':
                                        # Check if it looks like a dividend yield (reasonable percentage)
                                        try:
                                            yield_value = float(re.sub(r'[^\d.]', '', cell_text))
                                            if 0 < yield_value < 50:  # Reasonable dividend yield range
                                                dividend_yield = cell_text
                                                break
                                        except:
                                            continue
                            except:
                                pass
                            
                            stock_data = {
                                'symbol': symbol,
                                'company_name': company_name,
                                'price': price,
                                'change': change,
                                'change_percent': change_percent,
                                'volume': volume,
                                'market_cap': market_cap,
                                'dividend_yield': dividend_yield
                            }
                            
                            stocks_data.append(stock_data)
                            
                            # Print progress every 100 stocks
                            if (i + 1) % 100 == 0:
                                print(f"Processed {i + 1} stocks...")
                            
                    except Exception as e:
                        print(f"Error processing row {i}: {e}")
                        continue
                
                print(f"Successfully extracted {len(stocks_data)} stocks")
                
                if stocks_data:
                    return stocks_data
                else:
                    print("No stock data found, retrying...")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached.")
                    
            finally:
                if driver:
                    driver.quit()
                    
        return []
    
    def scrape_requests_fallback(self, url):
        """
        Fallback method using requests for static content
        
        Args:
            url (str): URL to scrape
            
        Returns:
            list: List of dictionaries containing stock data
        """
        try:
            print("Trying fallback method with requests...")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for stock data in the page
            stocks_data = []
            
            # Try to find table rows with stock data
            rows = soup.find_all('tr', {'data-rowkey': True})
            
            if not rows:
                # Alternative: look for any table rows in the content
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    if rows:
                        break
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:  # Minimum columns needed
                    # Extract basic information
                    symbol = cells[0].get_text(strip=True)
                    price = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    
                    # Try to find dividend yield in later columns
                    dividend_yield = ""
                    for cell in cells[2:]:
                        text = cell.get_text(strip=True)
                        if '%' in text and text != '—':
                            dividend_yield = text
                            break
                    
                    stocks_data.append({
                        'symbol': symbol,
                        'company_name': '',
                        'price': price,
                        'change': '',
                        'change_percent': '',
                        'volume': '',
                        'market_cap': '',
                        'dividend_yield': dividend_yield
                    })
            
            return stocks_data
            
        except Exception as e:
            print(f"Fallback method failed: {e}")
            return []
    
    def scrape_all_stocks(self, max_load_clicks=5):
        """
        Scrape all US stocks from the market movers page with load more functionality
        
        Args:
            max_load_clicks (int): Maximum number of times to click load more
            
        Returns:
            list: List of dictionaries containing stock data
        """
        all_stocks_url = "https://www.tradingview.com/markets/stocks-usa/market-movers-all-stocks/"
        
        print(f"Scraping with up to {max_load_clicks} load more clicks...")
        
        # Try Selenium first with load more functionality
        stocks_data = self.scrape_with_selenium(all_stocks_url, max_load_clicks=max_load_clicks)
        
        # If Selenium fails, try fallback method
        if not stocks_data:
            print("Selenium method failed, trying fallback...")
            stocks_data = self.scrape_requests_fallback(all_stocks_url)
        
        return stocks_data
    
    def save_to_csv(self, data, filename="tradingview_stocks.csv"):
        """
        Save scraped data to CSV file
        
        Args:
            data (list): List of dictionaries containing stock data
            filename (str): Output filename
        """
        if data:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            print(f"Saved {len(data)} stocks to CSV")
            return df
        else:
            print("No data to save")
            return None
    
    def filter_dividend_stocks(self, data, min_yield=0.0):
        """
        Filter stocks by minimum dividend yield
        
        Args:
            data (list): List of stock dictionaries
            min_yield (float): Minimum dividend yield percentage
            
        Returns:
            list: Filtered list of stocks with dividends
        """
        dividend_stocks = []
        
        for stock in data:
            dividend_text = stock.get('dividend_yield', '')
            if dividend_text and dividend_text != '—' and dividend_text != '':
                try:
                    # Extract numeric value from dividend yield
                    dividend_value = float(re.sub(r'[^\d.]', '', dividend_text))
                    if dividend_value >= min_yield:
                        dividend_stocks.append(stock)
                except (ValueError, TypeError):
                    continue
        
        return dividend_stocks
    
    def get_summary_stats(self, data):
        """
        Get summary statistics of the scraped data
        
        Args:
            data (list): List of stock dictionaries
            
        Returns:
            dict: Summary statistics
        """
        if not data:
            return {}
        
        total_stocks = len(data)
        stocks_with_dividends = len([s for s in data if s.get('dividend_yield') and s['dividend_yield'] != '—'])
        stocks_with_prices = len([s for s in data if s.get('price') and s['price'] != '—'])
        
        return {
            'total_stocks': total_stocks,
            'stocks_with_dividends': stocks_with_dividends,
            'stocks_with_prices': stocks_with_prices,
            'dividend_percentage': (stocks_with_dividends / total_stocks * 100) if total_stocks > 0 else 0
        }

# Usage example
def main():
    """Main function to demonstrate the scraper with load more functionality"""
    scraper = TradingViewScraper(headless=False)  # Set to True for headless mode
    
    print("=== TradingView Stock Scraper with Load More ===")
    print("This scraper will automatically click 'Load More' to get additional stock data\n")
    
    # Scrape all stocks with load more functionality
    print("Scraping all stocks with load more functionality...")
    max_load_clicks = 5  # Adjust this number based on how much data you want (Make this an insertable data)
    
    all_stocks = scraper.scrape_all_stocks(max_load_clicks=max_load_clicks)
    
    if all_stocks:
        # Get summary statistics
        stats = scraper.get_summary_stats(all_stocks)
        print(f"\n=== SCRAPING RESULTS ===")
        print(f"Total stocks found: {stats['total_stocks']}")
        print(f"Stocks with dividend data: {stats['stocks_with_dividends']}")
        print(f"Stocks with price data: {stats['stocks_with_prices']}")
        print(f"Dividend coverage: {stats['dividend_percentage']:.1f}%")
        
        # Save all stocks data
        df_all = scraper.save_to_csv(all_stocks, "all_stocks_with_loadmore.csv")
        
        # Filter and save dividend-paying stocks
        dividend_stocks = scraper.filter_dividend_stocks(all_stocks, min_yield=1.0)
        
        if dividend_stocks:
            print(f"\nFound {len(dividend_stocks)} dividend-paying stocks (>1% yield)")
            df_dividends = scraper.save_to_csv(dividend_stocks, "dividend_stocks_with_loadmore.csv")
            
            print("\n=== SAMPLE DIVIDEND STOCKS ===")
            for i, stock in enumerate(dividend_stocks[:10]):
                print(f"{i+1:2d}. {stock['symbol']:6s} - {stock['company_name'][:30]:30s} - Price: {stock['price']:>8s} - Dividend: {stock['dividend_yield']:>6s}")
        else:
            print("\nNo dividend-paying stocks found with the specified criteria")
        
        # Show sample of all stocks
        print(f"\n=== SAMPLE OF ALL STOCKS (First 10) ===")
        for i, stock in enumerate(all_stocks[:10]):
            print(f"{i+1:2d}. {stock['symbol']:6s} - {stock['company_name'][:30]:30s} - Price: {stock['price']:>8s}")
    
    else:
        print("No stock data was scraped. Please check the website structure or connection.")
    
    print("\n=== SCRAPING COMPLETED ===")

if __name__ == "__main__":
    main()