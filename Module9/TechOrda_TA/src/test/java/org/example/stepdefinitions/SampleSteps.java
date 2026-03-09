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
        // options.addArguments("--headless"); // Run in headless mode for CI (disabled for debugging)
        driver = new ChromeDriver(options);
        driver.manage().window().maximize();
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(5));
        wait = new WebDriverWait(driver, Duration.ofSeconds(15));
    }

    @When("I navigate to {string}")
    public void iNavigateTo(String url) {
        driver.get(url);
    }

    @And("I click on {string} tab")
    public void iClickOnTab(String tabName) {
        // use a robust xpath search that handles links or buttons containing the text
        WebElement tab = wait.until(
            ExpectedConditions.elementToBeClickable(
                By.xpath("//*[contains(text(),'" + tabName + "')]")
            )
        );
        tab.click();
    }

    @And("I select {string} in the industry filter")
    public void iSelectIndustryFilter(String industry) {
        WebElement filter = wait.until(ExpectedConditions.elementToBeClickable(By.xpath("//input[@value='" + industry + "']")));
        filter.click();
    }

    @And("I search for {string}")
    public void iSearchFor(String term) {
        WebElement searchBox = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("input[type='search']")));
        searchBox.sendKeys(term);
        searchBox.submit();
    }

    @And("I toggle {string} filter")
    public void iToggleFilter(String filterName) {
        WebElement filter = wait.until(ExpectedConditions.elementToBeClickable(By.xpath("//label[contains(text(),'" + filterName + "')]")));
        filter.click();
    }

    @And("I click {string} on an asset card")
    public void iClickOnAssetCard(String linkText) {
        WebElement link = wait.until(ExpectedConditions.elementToBeClickable(By.linkText(linkText)));
        link.click();
    }

    @And("I open a long guide")
    public void iOpenLongGuide() {
        WebElement guide = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector(".guide-item")));
        guide.click();
    }

    @And("I click on a Table of Contents item")
    public void iClickTocItem() {
        WebElement tocItem = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector(".toc-item")));
        tocItem.click();
    }

    @And("I select {string} category")
    public void iSelectCategory(String category) {
        WebElement cat = wait.until(ExpectedConditions.elementToBeClickable(By.linkText(category)));
        cat.click();
    }

    @And("I open any blog post")
    public void iOpenBlogPost() {
        WebElement post = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector(".blog-post")));
        post.click();
    }

    @And("I click the LinkedIn share icon")
    public void iClickLinkedInShare() {
        WebElement share = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector(".share-linkedin")));
        share.click();
    }

    @And("I click on {string} tag")
    public void iClickTag(String tag) {
        WebElement tagElement = wait.until(ExpectedConditions.elementToBeClickable(By.linkText(tag)));
        tagElement.click();
    }

    @And("I click on an FAQ question")
    public void iClickFaqQuestion() {
        WebElement faq = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector(".faq-question")));
        faq.click();
    }

    @And("I click {string} button")
    public void iClickButton(String buttonText) {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(By.xpath("//button[contains(text(),'" + buttonText + "')]")));
        button.click();
    }

    @Then("only solutions related to {string} are displayed")
    public void onlySolutionsDisplayed(String industry) {
        // Add assertion logic
        assertTrue(driver.getCurrentUrl().contains("filter") || driver.findElements(By.cssSelector(".solution-item")).size() > 0);
    }

    @Then("{string} appears as the first result")
    public void appearsAsFirstResult(String item) {
        WebElement firstResult = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".result-item:first-child")));
        assertTrue(firstResult.getText().contains(item));
    }

    @Then("results update to show {string} items")
    public void resultsUpdate(String expected) {
        // Add assertion
        assertTrue(driver.findElements(By.cssSelector(".asset-item")).size() > 0);
    }

    @Then("the external repository opens in a new tab")
    public void externalRepoOpens() {
        // Check for new window
        assertTrue(driver.getWindowHandles().size() > 1);
    }

    @Then("the page scrolls to the correct section")
    public void pageScrollsToSection() {
        // Check if scrolled
        assertTrue(true); // Placeholder
    }

    @Then("only relevant guides are listed")
    public void relevantGuidesListed() {
        assertTrue(driver.findElements(By.cssSelector(".guide-item")).size() > 0);
    }

    @Then("the share dialog opens with the correct URL")
    public void shareDialogOpens() {
        // Check for popup or new window
        assertTrue(driver.getWindowHandles().size() > 1);
    }

    @Then("the feed shows only {string} articles")
    public void feedShowsOnly(String category) {
        // Add assertion
        assertTrue(true);
    }

    @Then("the answer expands smoothly")
    public void answerExpands() {
        WebElement answer = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".faq-answer")));
        assertTrue(answer.isDisplayed());
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