import { test, expect } from "@playwright/test";

test("sample lecture can be searched and source evidence is shown", async ({ page }) => {
  await page.goto("/?e2e=1");

  await expect(page.locator("h1")).toContainText(/replay the whole lecture/i);
  await expect(page.getByText("Regression models")).toHaveCount(0);

  await page.getByRole("button", { name: "Load sample lecture" }).click();

  await expect(page.locator("#timeline")).toBeVisible();
  await expect(page.locator(".timeline-item")).toHaveCount(8);

  await page.locator("#question").fill("What is a base case?");
  await page.locator("#searchBtn").click();

  const first = page.locator(".result-card").first();
  await expect(first).toContainText(/base case/i);
  await expect(first.getByRole("button", { name: /Source/ })).toBeVisible();
});

test("saved lectures persist in IndexedDB after reload", async ({ page }) => {
  await page.goto("/?e2e=1");
  await page.getByRole("button", { name: "Load sample lecture" }).click();
  await page.locator("#saveLecture").click();

  await expect(page.locator("#libraryList")).toContainText("Recursion & call stack");

  await page.reload();
  await expect(page.locator("#libraryList")).toContainText("Recursion & call stack");

  await page.locator("#libraryList [data-open]").first().click();
  await expect(page.locator("#lectureTitle")).toHaveValue("Recursion & call stack");
  await expect(page.locator("#timeline")).toBeVisible();
});

test("new lecture resets the workspace without fake future classes", async ({ page }) => {
  await page.goto("/?e2e=1");
  await page.getByRole("button", { name: "Load sample lecture" }).click();
  await page.locator("#newLecture").click();

  await expect(page.locator("#lectureTitle")).toHaveValue("Untitled lecture");
  await expect(page.locator("#transcriptInput")).toHaveValue("");
  await expect(page.getByText("Regression models")).toHaveCount(0);
});
