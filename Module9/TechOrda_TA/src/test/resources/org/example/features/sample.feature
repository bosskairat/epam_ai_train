Feature: EPAM SolutionsHub Testing

  Scenario: SOL-01 Filter by Industry
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Solutions" tab
    And I select "Financial Services" in the industry filter
    Then only solutions related to Financial Services are displayed

  Scenario: SOL-02 Exact Search
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Solutions" tab
    And I search for "ReportPortal"
    Then ReportPortal appears as the first result

  Scenario: AST-01 Filter Toggle
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Assets" tab
    And I toggle "Intelligent Automation" filter
    Then results update to show EPAM Intelligent Automation items

  Scenario: AST-02 GitHub Redirect
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Assets" tab
    And I click "View on GitHub" on an asset card
    Then the external repository opens in a new tab

  Scenario: GUI-01 Anchor Link Jump
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Guides" tab
    And I open a long guide
    And I click on a Table of Contents item
    Then the page scrolls to the correct section

  Scenario: GUI-02 Category Selection
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Guides" tab
    And I select "Solution Owners" category
    Then only relevant guides are listed

  Scenario: BLG-01 Social Share
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Blog" tab
    And I open any blog post
    And I click the LinkedIn share icon
    Then the share dialog opens with the correct URL

  Scenario: BLG-02 Tag Filtering
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "Blog" tab
    And I click on "Insights" tag
    Then the feed shows only "Insights" articles

  Scenario: ABT-01 FAQ Interaction
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "About" tab
    And I click on an FAQ question
    Then the answer expands smoothly

  Scenario: ABT-02 Main CTA Function
    Given I open the browser
    When I navigate to "https://solutionshub.epam.com/"
    And I click on "About" tab
    And I click "Get Started" button
    Then I am navigated to the Solutions catalog
