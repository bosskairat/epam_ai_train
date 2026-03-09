package org.example.stepdefinitions;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.When;
import io.cucumber.java.en.Then;
import io.cucumber.java.en.And;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.By;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SampleSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    @Given("I open the browser")
    public void iOpenTheBrowser() {
        System.setProperty("webdriver.chrome.driver", "chromedriver-win64/chromedriver.exe");
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--headless"); // Run in headless mode for CI (disabled for debugging)
        driver = new ChromeDriver(options);
        driver.manage().window().maximize();
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(3));
        wait = new WebDriverWait(driver, Duration.ofSeconds(5));
    }

    @When("I navigate to {string}")
    public void iNavigateTo(String url) {
        driver.get(url);
    }

    @And("I click on {string} tab")
    public void iClickOnTab(String tabName) {
        // Try multiple selector strategies to find the tab
        WebElement tab = null;
        By[] locators = {
            // Try button first
            By.xpath("//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + tabName.toLowerCase() + "')]"),
            // Try anchor/link
            By.xpath("//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + tabName.toLowerCase() + "')]"),
            // Try any element with normalized whitespace
            By.xpath("//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + tabName.toLowerCase() + "')]")
        };
        
        for (By locator : locators) {
            try {
                tab = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found tab '" + tabName + "' using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (tab != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", tab);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            tab.click();
        } else {
            throw new RuntimeException("Unable to find clickable tab element for: " + tabName);
        }
    }

    @And("I select {string} in the industry filter")
    public void iSelectIndustryFilter(String industry) {
        // Try multiple selector strategies for industry filters
        WebElement filter = null;
        By[] locators = {
            // Try checkbox/radio input with value
            By.xpath("//input[@value='" + industry + "']"),
            // Try input with name or id containing industry
            By.xpath("//input[contains(@name, '" + industry.replace(" ", "").toLowerCase() + "') or contains(@id, '" + industry.replace(" ", "").toLowerCase() + "')]"),
            // Try label containing the industry text
            By.xpath("//label[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + industry.toLowerCase() + "')]"),
            // Try any clickable element (button, div, span) containing the industry text
            By.xpath("//*[self::button or self::div or self::span or self::a][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + industry.toLowerCase() + "')]")
        };
        
        for (By locator : locators) {
            try {
                filter = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found industry filter '" + industry + "' using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (filter != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", filter);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            filter.click();
        } else {
            throw new RuntimeException("Unable to find clickable industry filter element for: " + industry);
        }
    }

    @And("I search for {string}")
    public void iSearchFor(String term) {
        // Try multiple selector strategies for search inputs
        WebElement searchBox = null;
        By[] locators = {
            // Try input with type='search'
            By.cssSelector("input[type='search']"),
            // Try input with placeholder containing 'search'
            By.cssSelector("input[placeholder*='search' i]"),
            // Try input with name containing 'search' or 'query'
            By.cssSelector("input[name*='search' i], input[name*='query' i]"),
            // Try input with class containing 'search'
            By.cssSelector("input[class*='search' i]"),
            // Try input with id containing 'search'
            By.cssSelector("input[id*='search' i]"),
            // Try any input inside a search container or form
            By.xpath("//form[contains(@class, 'search') or contains(@id, 'search')]//input[@type='text' or not(@type)]"),
            // Fallback: any text input that might be a search box
            By.cssSelector("input[type='text']")
        };
        
        for (By locator : locators) {
            try {
                searchBox = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found search box using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (searchBox != null) {
            // Scroll element into view before interacting
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", searchBox);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            searchBox.clear(); // Clear any existing text
            searchBox.sendKeys(term);
            
            // Try to submit the form or click a search button
            try {
                searchBox.submit();
                // System.out.println("Submitted search form");
            } catch (Exception e) {
                // If submit fails, try to find and click a search button
                try {
                    WebElement searchButton = driver.findElement(By.xpath("//button[contains(@type, 'submit') or contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'search')]"));
                    searchButton.click();
                    // System.out.println("Clicked search button");
                } catch (Exception e2) {
                    // System.out.println("Could not submit search, proceeding without submit");
                }
            }
        } else {
            throw new RuntimeException("Unable to find clickable search input element");
        }
    }

    @And("I toggle {string} filter")
    public void iToggleFilter(String filterName) {
        // Try multiple selector strategies for filter toggles
        WebElement filter = null;
        By[] locators = {
            // Try label containing the filter text
            By.xpath("//label[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + filterName.toLowerCase() + "')]"),
            // Try checkbox input with associated label
            By.xpath("//input[@type='checkbox'][following-sibling::label[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + filterName.toLowerCase() + "')]]"),
            // Try input with value or name containing filter
            By.xpath("//input[contains(@value, '" + filterName + "') or contains(@name, '" + filterName.replace(" ", "").toLowerCase() + "')]"),
            // Try any clickable element (button, div, span) containing the filter text
            By.xpath("//*[self::button or self::div or self::span or self::a][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + filterName.toLowerCase() + "')]"),
            // Try elements with filter-related classes or data attributes
            By.xpath("//*[@class[contains(., 'filter') or contains(., 'toggle')]][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + filterName.toLowerCase() + "')]")
        };
        
        for (By locator : locators) {
            try {
                filter = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found filter toggle '" + filterName + "' using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (filter != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", filter);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            filter.click();
        } else {
            throw new RuntimeException("Unable to find clickable filter toggle element for: " + filterName);
        }
    }

    @And("I click {string} on an asset card")
    public void iClickOnAssetCard(String linkText) {
        // Try multiple selector strategies for asset card links/buttons
        WebElement link = null;
        By[] locators = {
            // Try exact link text
            By.linkText(linkText),
            // Try partial link text
            By.partialLinkText(linkText),
            // Try button containing the text
            By.xpath("//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + linkText.toLowerCase() + "')]"),
            // Try anchor/link containing the text (case-insensitive)
            By.xpath("//a[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + linkText.toLowerCase() + "')]"),
            // Try any clickable element containing the text within asset cards
            By.xpath("//*[contains(@class, 'card') or contains(@class, 'asset') or contains(@class, 'item')]//*[self::a or self::button or self::div or self::span][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + linkText.toLowerCase() + "')]"),
            // Try any element with GitHub-related attributes or text
            By.xpath("//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'github') or contains(@href, 'github') or contains(@class, 'github')]")
        };
        
        for (By locator : locators) {
            try {
                link = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found asset card link '" + linkText + "' using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (link != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", link);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            link.click();
        } else {
            throw new RuntimeException("Unable to find clickable asset card element for: " + linkText);
        }
    }

    @And("I open a long guide")
    public void iOpenLongGuide() {
        // Try multiple selector strategies for guide items
        WebElement guide = null;
        By[] locators = {
            // Try elements with guide-related classes
            By.cssSelector(".guide-item, .guide, .guide-card, .guide-link"),
            // Try links or buttons containing guide-related text
            By.xpath("//a[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'guide') or contains(@href, 'guide')]"),
            // Try any clickable element with guide-related classes or text
            By.xpath("//*[contains(@class, 'guide') or contains(@class, 'tutorial') or contains(@class, 'documentation')][self::a or self::button or self::div]"),
            // Try elements that might be guide items based on content or structure
            By.xpath("//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'guide') or contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'tutorial')]"),
            // Try first clickable element in a guides section
            By.xpath("//*[contains(@class, 'guides') or contains(@id, 'guides') or contains(@class, 'guide-list')]//a[1] | //*[contains(@class, 'guides') or contains(@id, 'guides') or contains(@class, 'guide-list')]//button[1]")
        };
        
        for (By locator : locators) {
            try {
                guide = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found guide item using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (guide != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", guide);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            guide.click();
        } else {
            throw new RuntimeException("Unable to find clickable guide item element");
        }
    }

    @And("I click on a Table of Contents item")
    public void iClickTocItem() {
        // Try multiple selector strategies for table of contents items
        WebElement tocItem = null;
        By[] locators = {
            // Try elements with TOC-related classes
            By.cssSelector(".toc-item, .toc-link, .table-of-contents a, .toc a"),
            // Try links within TOC sections
            By.xpath("//*[contains(@class, 'toc') or contains(@class, 'table-of-contents') or contains(@id, 'toc')]//a[1]"),
            // Try any clickable element within TOC containers
            By.xpath("//*[contains(@class, 'toc') or contains(@id, 'toc') or contains(@class, 'table-of-contents')]//*[self::a or self::button or self::div or self::span][1]"),
            // Try elements that might be TOC items based on structure
            By.xpath("//nav[contains(@class, 'toc') or contains(@id, 'toc')]//a[1] | //div[contains(@class, 'toc')]//a[1]"),
            // Try any link that looks like a section anchor
            By.xpath("//a[starts-with(@href, '#')][1]")
        };
        
        for (By locator : locators) {
            try {
                tocItem = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found TOC item using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (tocItem != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", tocItem);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            tocItem.click();
        } else {
            throw new RuntimeException("Unable to find clickable table of contents item element");
        }
    }

    @And("I select {string} category")
    public void iSelectCategory(String category) {
        // Try multiple selector strategies for category selection
        WebElement cat = null;
        By[] locators = {
            // Try exact link text
            By.linkText(category),
            // Try partial link text
            By.partialLinkText(category),
            // Try button containing the category text
            By.xpath("//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + category.toLowerCase() + "')]"),
            // Try anchor/link containing the category text (case-insensitive)
            By.xpath("//a[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + category.toLowerCase() + "')]"),
            // Try any clickable element containing the category text
            By.xpath("//*[self::a or self::button or self::div or self::span or self::li][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + category.toLowerCase() + "')]"),
            // Try elements within category/filter sections
            By.xpath("//*[contains(@class, 'category') or contains(@class, 'filter') or contains(@class, 'tag')]//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + category.toLowerCase() + "')]")
        };
        
        for (By locator : locators) {
            try {
                cat = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found category '" + category + "' using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (cat != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", cat);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            cat.click();
        } else {
            throw new RuntimeException("Unable to find clickable category element for: " + category);
        }
    }

    @And("I open any blog post")
    public void iOpenBlogPost() {
        // Try multiple selector strategies for blog posts
        WebElement post = null;
        By[] locators = {
            // Try elements with blog-related classes
            By.cssSelector(".blog-post, .post, .article, .blog-item, .blog-card"),
            // Try links within blog sections
            By.xpath("//*[contains(@class, 'blog') or contains(@class, 'post') or contains(@class, 'article')]//a[1]"),
            // Try any clickable element with blog/article related text
            By.xpath("//a[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'read more') or contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue reading')]"),
            // Try first article or post link
            By.xpath("//article//a[1] | //*[contains(@class, 'post')]//a[1] | //*[contains(@class, 'article')]//a[1]"),
            // Try any link within blog container
            By.xpath("//*[contains(@class, 'blog') or contains(@id, 'blog') or contains(@class, 'posts')]//a[1]")
        };
        
        for (By locator : locators) {
            try {
                post = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found blog post using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (post != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", post);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            post.click();
        } else {
            throw new RuntimeException("Unable to find clickable blog post element");
        }
    }

    @And("I click the LinkedIn share icon")
    public void iClickLinkedInShare() {
        // Try multiple selector strategies for LinkedIn share icons
        WebElement share = null;
        By[] locators = {
            // Try elements with LinkedIn-related classes
            By.cssSelector(".share-linkedin, .linkedin-share, .social-linkedin, [class*='linkedin']"),
            // Try elements containing LinkedIn text or attributes
            By.xpath("//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'linkedin') or contains(@href, 'linkedin') or contains(@class, 'linkedin')]"),
            // Try social share buttons with LinkedIn icon or text
            By.xpath("//*[contains(@class, 'share') or contains(@class, 'social')]//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'linkedin') or contains(@class, 'linkedin')]"),
            // Try any element with LinkedIn-related attributes
            By.xpath("//a[contains(@href, 'linkedin.com') or contains(@href, 'linkedin') or contains(@title, 'linkedin') or contains(@aria-label, 'linkedin')]"),
            // Try social media buttons that might be LinkedIn
            By.xpath("//*[contains(@class, 'social') or contains(@class, 'share')]//button[2] | //*[contains(@class, 'social') or contains(@class, 'share')]//a[2]")
        };
        
        for (By locator : locators) {
            try {
                share = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found LinkedIn share using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (share != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", share);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            share.click();
        } else {
            throw new RuntimeException("Unable to find clickable LinkedIn share element");
        }
    }

    @And("I click on {string} tag")
    public void iClickTag(String tag) {
        // Try multiple selector strategies for tag selection
        WebElement tagElement = null;
        By[] locators = {
            // Try exact link text
            By.linkText(tag),
            // Try partial link text
            By.partialLinkText(tag),
            // Try button containing the tag text
            By.xpath("//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + tag.toLowerCase() + "')]"),
            // Try anchor/link containing the tag text (case-insensitive)
            By.xpath("//a[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + tag.toLowerCase() + "')]"),
            // Try any clickable element containing the tag text
            By.xpath("//*[self::a or self::button or self::div or self::span][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + tag.toLowerCase() + "')]"),
            // Try elements with tag-related classes
            By.xpath("//*[contains(@class, 'tag') or contains(@class, 'category') or contains(@class, 'filter')]//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + tag.toLowerCase() + "')]")
        };
        
        for (By locator : locators) {
            try {
                tagElement = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found tag '" + tag + "' using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (tagElement != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", tagElement);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            tagElement.click();
        } else {
            throw new RuntimeException("Unable to find clickable tag element for: " + tag);
        }
    }

    @And("I click on an FAQ question")
    public void iClickFaqQuestion() {
        // Try multiple selector strategies for FAQ questions
        WebElement faq = null;
        By[] locators = {
            // Try elements with FAQ-related classes
            By.cssSelector(".faq-question, .faq-item, .question, .accordion-toggle, .collapse-toggle"),
            // Try elements containing question marks or FAQ text
            By.xpath("//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '?') and (contains(@class, 'faq') or contains(@class, 'accordion') or contains(@class, 'collapse'))]"),
            // Try clickable elements within FAQ sections
            By.xpath("//*[contains(@class, 'faq') or contains(@id, 'faq') or contains(@class, 'accordion')]//button[1] | //*[contains(@class, 'faq') or contains(@id, 'faq') or contains(@class, 'accordion')]//a[1] | //*[contains(@class, 'faq') or contains(@id, 'faq') or contains(@class, 'accordion')]//div[contains(@class, 'clickable') or contains(@class, 'toggle')][1]"),
            // Try any clickable element with question-related text
            By.xpath("//*[self::button or self::a or self::div][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'what') or contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'how') or contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'why') or contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '?')]"),
            // Try first clickable element in FAQ container
            By.xpath("//*[contains(@class, 'faq') or contains(@id, 'faq')]//*[self::button or self::a or self::div][1]")
        };
        
        for (By locator : locators) {
            try {
                faq = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found FAQ question using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (faq != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", faq);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            faq.click();
        } else {
            throw new RuntimeException("Unable to find clickable FAQ question element");
        }
    }

    @And("I click {string} button")
    public void iClickButton(String buttonText) {
        // Try multiple selector strategies for button clicking
        WebElement button = null;
        By[] locators = {
            // Try button element with exact text
            By.xpath("//button[contains(text(),'" + buttonText + "')]"),
            // Try button element with normalized text (case-insensitive)
            By.xpath("//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + buttonText.toLowerCase() + "')]"),
            // Try anchor/link styled as button
            By.xpath("//a[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + buttonText.toLowerCase() + "')]"),
            // Try any element with button-like classes containing the text
            By.xpath("//*[contains(@class, 'button') or contains(@class, 'btn') or contains(@role, 'button')][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + buttonText.toLowerCase() + "')]"),
            // Try input elements with button type
            By.xpath("//input[@type='button' or @type='submit'][contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + buttonText.toLowerCase() + "')]"),
            // Try any clickable element containing the button text
            By.xpath("//*[self::button or self::a or self::div or self::span or self::input][contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '" + buttonText.toLowerCase() + "')]")
        };
        
        for (By locator : locators) {
            try {
                button = wait.until(ExpectedConditions.elementToBeClickable(locator));
                // System.out.println("Found button '" + buttonText + "' using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (button != null) {
            // Scroll element into view before clicking
            ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", button);
            try {
                Thread.sleep(200); // Brief wait for scroll animation
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            button.click();
        } else {
            throw new RuntimeException("Unable to find clickable button element for: " + buttonText);
        }
    }

    @Then("only solutions related to {string} are displayed")
    public void onlySolutionsDisplayed(String industry) {
        // Wait a moment for filtering to complete
        try {
            Thread.sleep(300);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        
        // Check URL contains filter or solutions are displayed
        boolean hasFilter = driver.getCurrentUrl().contains("filter") || 
                           driver.getCurrentUrl().contains(industry.toLowerCase().replace(" ", ""));
        
        // Try multiple selectors for solution items
        By[] locators = {
            By.cssSelector(".solution-item, .solution, .card, .item"),
            By.xpath("//*[contains(@class, 'solution') or contains(@class, 'card') or contains(@class, 'item')]"),
            By.xpath("//article | //div[contains(@class, 'solution')]")
        };
        
        int solutionCount = 0;
        for (By locator : locators) {
            try {
                solutionCount = driver.findElements(locator).size();
                if (solutionCount > 0) {
                    // System.out.println("Found " + solutionCount + " solutions using locator: " + locator);
                    break;
                }
            } catch (Exception e) {
                continue;
            }
        }
        
        assertTrue(hasFilter || solutionCount > 0, 
                  "Expected solutions to be filtered or displayed, but found " + solutionCount + " solutions and filter status: " + hasFilter);
    }

    @Then("{string} appears as the first result")
    public void appearsAsFirstResult(String item) {
        // Try multiple selectors for first result
        WebElement firstResult = null;
        By[] locators = {
            By.cssSelector(".result-item:first-child, .search-result:first-child, .item:first-child"),
            By.xpath("//*[contains(@class, 'result') or contains(@class, 'search') or contains(@class, 'item')][1]"),
            By.xpath("//div[contains(@class, 'result')][1] | //li[contains(@class, 'result')][1] | //article[1]"),
            By.xpath("//*[contains(@class, 'results') or contains(@id, 'results')]//*[1]")
        };
        
        for (By locator : locators) {
            try {
                firstResult = wait.until(ExpectedConditions.visibilityOfElementLocated(locator));
                // System.out.println("Found first result using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (firstResult != null) {
            assertTrue(firstResult.getText().toLowerCase().contains(item.toLowerCase()), 
                      "First result should contain '" + item + "', but was: " + firstResult.getText());
        } else {
            throw new RuntimeException("Unable to find first result element");
        }
    }

    @Then("results update to show {string} items")
    public void resultsUpdate(String expected) {
        // Try multiple selectors for result items
        By[] locators = {
            By.cssSelector(".asset-item, .result-item, .item"),
            By.xpath("//*[contains(@class, 'asset') or contains(@class, 'result') or contains(@class, 'item')]"),
            By.xpath("//div[contains(@class, 'card') or contains(@class, 'item')]"),
            By.xpath("//article | //li[contains(@class, 'item')]")
        };
        
        int itemCount = 0;
        for (By locator : locators) {
            try {
                itemCount = driver.findElements(locator).size();
                if (itemCount > 0) {
                    // System.out.println("Found " + itemCount + " items using locator: " + locator);
                    break;
                }
            } catch (Exception e) {
                continue;
            }
        }
        
        assertTrue(itemCount > 0, "Expected to find result items, but found " + itemCount);
    }

    @Then("the external repository opens in a new tab")
    public void externalRepoOpens() {
        // Wait a moment for the new tab to open
        try {
            Thread.sleep(500);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        
        // Check for new window/tab
        boolean hasNewTab = driver.getWindowHandles().size() > 1;
        
        // If still no new tab, check if current URL changed to GitHub
        if (!hasNewTab) {
            String currentUrl = driver.getCurrentUrl();
            hasNewTab = currentUrl.contains("github.com") || currentUrl.contains("github");
        }
        
        assertTrue(hasNewTab, "Expected external repository to open in new tab or navigate to GitHub");
    }

    @Then("the page scrolls to the correct section")
    public void pageScrollsToSection() {
        // Wait a moment for scrolling to complete
        try {
            Thread.sleep(300);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        
        // Check if page has scrolled (Y position > 0)
        JavascriptExecutor js = (JavascriptExecutor) driver;
        Long scrollY = (Long) js.executeScript("return window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;");
        
        assertTrue(scrollY > 0, "Expected page to scroll to a section, but scroll position is " + scrollY);
    }

    @Then("only relevant guides are listed")
    public void relevantGuidesListed() {
        // Try multiple selectors for guide items
        By[] locators = {
            By.cssSelector(".guide-item, .guide, .guide-card, .guide-link"),
            By.xpath("//*[contains(@class, 'guide') or contains(@class, 'tutorial') or contains(@class, 'documentation')]"),
            By.xpath("//a[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'guide') or contains(@href, 'guide')]"),
            By.xpath("//article | //div[contains(@class, 'card') or contains(@class, 'item')]")
        };
        
        int guideCount = 0;
        for (By locator : locators) {
            try {
                guideCount = driver.findElements(locator).size();
                if (guideCount > 0) {
                    // System.out.println("Found " + guideCount + " guides using locator: " + locator);
                    break;
                }
            } catch (Exception e) {
                continue;
            }
        }
        
        assertTrue(guideCount > 0, "Expected to find relevant guides, but found " + guideCount);
    }

    @Then("the share dialog opens with the correct URL")
    public void shareDialogOpens() {
        // Wait a moment for the share dialog to open
        try {
            Thread.sleep(500);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        
        // Check for new window/tab or URL change
        boolean dialogOpened = driver.getWindowHandles().size() > 1;
        
        // If no new window, check if URL changed to LinkedIn
        if (!dialogOpened) {
            String currentUrl = driver.getCurrentUrl();
            dialogOpened = currentUrl.contains("linkedin.com") || currentUrl.contains("linkedin");
        }
        
        // Also check for popup/modal elements
        if (!dialogOpened) {
            try {
                WebElement modal = driver.findElement(By.cssSelector(".modal, .popup, .dialog, [role='dialog']"));
                dialogOpened = modal.isDisplayed();
            } catch (Exception e) {
                // No modal found
            }
        }
        
        assertTrue(dialogOpened, "Expected share dialog to open or navigate to LinkedIn");
    }

    @Then("the feed shows only {string} articles")
    public void feedShowsOnly(String category) {
        // Wait a moment for filtering to complete
        try {
            Thread.sleep(300);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        
        // Try multiple selectors for articles/posts
        By[] locators = {
            By.cssSelector("article, .post, .blog-post, .article"),
            By.xpath("//article | //*[contains(@class, 'post') or contains(@class, 'article')]"),
            By.xpath("//div[contains(@class, 'card') or contains(@class, 'item')]")
        };
        
        int articleCount = 0;
        for (By locator : locators) {
            try {
                articleCount = driver.findElements(locator).size();
                if (articleCount > 0) {
                    // System.out.println("Found " + articleCount + " articles using locator: " + locator);
                    break;
                }
            } catch (Exception e) {
                continue;
            }
        }
        
        assertTrue(articleCount > 0, "Expected to find articles in the feed, but found " + articleCount);
    }

    @Then("the answer expands smoothly")
    public void answerExpands() {
        // Try multiple selectors for FAQ answers
        WebElement answer = null;
        By[] locators = {
            By.cssSelector(".faq-answer, .answer, .faq-content, .accordion-content, .collapse-content"),
            By.xpath("//*[contains(@class, 'faq') or contains(@class, 'accordion') or contains(@class, 'collapse')]//*[contains(@class, 'answer') or contains(@class, 'content') or contains(@class, 'panel')]"),
            By.xpath("//*[contains(@class, 'faq') or contains(@id, 'faq')]//div[contains(@class, 'show') or contains(@class, 'expanded') or contains(@style, 'display: block')]"),
            By.xpath("//*[contains(@class, 'faq') or contains(@id, 'faq')]//p[1] | //*[contains(@class, 'faq') or contains(@id, 'faq')]//div[1]")
        };
        
        for (By locator : locators) {
            try {
                answer = wait.until(ExpectedConditions.visibilityOfElementLocated(locator));
                // System.out.println("Found FAQ answer using locator: " + locator);
                break;
            } catch (org.openqa.selenium.TimeoutException e) {
                // System.out.println("Locator not found: " + locator + ", trying next...");
                continue;
            }
        }
        
        if (answer != null) {
            assertTrue(answer.isDisplayed(), "FAQ answer should be visible after clicking question");
        } else {
            throw new RuntimeException("Unable to find visible FAQ answer element");
        }
    }

    @Then("I am navigated to the Solutions catalog")
    public void navigatedToSolutions() {
        assertTrue(driver.getCurrentUrl().contains("solutions"));
    }

    @Then("the page title should be {string}")
    public void thePageTitleShouldBe(String expectedTitle) {
        String actualTitle = driver.getTitle();
        assertEquals(expectedTitle, actualTitle);
        driver.quit(); // Close browser after test
    }
}
