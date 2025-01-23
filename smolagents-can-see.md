---
title: "We now support VLMs in smolagents!"
thumbnail: /blog/assets/smolagents-can-see/thumbnail.png
authors:
  - user: m-ric
  - user: merve
  - user: albertvillanova
---
# Our smolagents can now see!

> You hypocrite, first take the log out of your own eye, and then you will see clearly to take the speck out of your brother's eye. *Matthew 7, 3-5*

## TL;DR

We have added vision support to smolagents, which unlocks the use of vision language models in agentic pipelines natively. 

## Table of Contents

- [Overview](#overview)
- [How we gave sight to smolagents](#how-we-gave-sight-to-smolagents)
- [Building a vision-enabled web browser agent](#building-a-vision-enabled-web-browser-agent)
- [How to create a Web browsing agent with vision](#how-to-create-a-web-browsing-agent-with-vision)
- [Next Steps](#next-steps)

## Overview

In the agentic world, many capabilities are hidden behind a vision wall. A common example is web browsing: web pages feature rich visual content that you never fully recover by simply extracting their text, be it the relative position of objects, messages transmitted through color, specific icons… In this case, vision is a real superpower for agents. So we just added this capability to our [smolagents](https://github.com/huggingface/smolagents)!

Teaser of what this gives: an agentic browser that navigates the web in complete autonomy!

### VIDEO EXTRACT

## How we gave sight to smolagents

🤔 How do we want to pass images to agents? Passing an image can be done in two ways:

1. You can have images directly available to the agent at start. This is often the case for Document AI.
2. Sometimes,images  need to be added dynamically. A good example iswhen a web browser just performed an action, and needs to see the impact on its viewports. 

#### 1. Pass images once at agent start

For the case where we want to pass images at once, we added the possibility to pass a list of images to the agent in the `run` method: `agent.run("Describe these images:", images=[image_1, image_2])` .

These image inputs are then stored in the `task_images` attribute of `TaskStep` along with the prompt of the task that you'd like to accomplish. 

When running the agent, they will be passed to the model. This comes in handy with cases like taking actions based on long PDFs that include visual elements.

#### 2. Pass images at each step ⇒ use a callback

How to dynamically add images into the agent’s memory?

To find out, we first need to understand how our agents work.

All agents in `smolagents` are based on singular `MultiStepAgent` class, which is an abstraction of ReAct framework. On a basic level, this class performs actions on a cycle of following steps, where existing variables and knowledge is incorporated into the agent logs as follows: 

- **Initialization:** the system prompt is stored in a `SystemPromptStep`, and the user query is logged into a `TaskStep`.
- **ReAct Loop (While):**
    1. Use `agent.write_inner_memory_from_logs()` to write the agent logs into a list of LLM-readable [chat messages](https://huggingface.co/docs/transformers/en/chat_templating).
    2. Send these messages to a `Model` object to get its completion. Parse the completion to get the action (a JSON blob for `ToolCallingAgent`, a code snippet for `CodeAgent`).
    3. Execute the action and logs result into memory (an `ActionStep`).
    4. At the end of each step, run all callback functions defined in `agent.step_callbacks`.
        ⇒ This is where we added support to images: make a callback that logs images into memory!
        

The figure below details this process:

![image.png](Blog%20post%20Agents%20can%20see%201811384ebcac804a879cd3d81eb6b0e5/image.png)

As you can see, for use cases where images are dynamically retrieved (e.g. web browser agent), we support adding images to the model’s `ActionStep`, in attribute `step_log.observation_images`.

This can be done via a callback, which will be run at the end of each step.

Let's demonstrate how to make such a callback, and using it to build a web browser agent.👇👇

### Building a vision-enabled web browser agent

Let’s see how to perform the dynamic image logging using a callback, to make a vision-enabled web browser agent. `save_screenshot` is our callback here that will be ran at the end of each step.

```python
def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0) # Let JavaScript animations happen before taking the screenshots
    driver = helium.get_driver()
    current_step = step_log.step_number
    if driver is not None:
        for step_logs in agent.logs: # Remove previous screenshots from logs for lean processing
            if isinstance(step_log, ActionStep) and step_log.step_number <= current_step - 2:
                step_logs.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        step_log.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    step_log.observations = url_info if step_logs.observations is None else step_log.observations + "\n" + url_info
    return
```

The most important line here is when we add the image in our observations images: `step_log.observations_images = [image.copy()]` 

This callback accepts both the `step_log`, and the `agent` itself as arguments. Having `agent` as an input allows to perform deeper operations than just modifying the last logs.

### How to create a Web browsing agent with vision

We’re going to use [helium](https://github.com/helium).It provides browser automations based on `selenium` : this will be an easier way for our agent to manipulate webpages.

```bash
pip install "smolagents[all]" helium selenium python-dotenv
```

The agent itself can use helium directly, so no need for specific tools: it can directly use helium to perform actions, such as `click(”top 10”)`  to click the button named “top 10” visible on the page. 

We still have to make two more tools to help the agent navigate the web: a tool to go back to the previous page, and another tool to close pop-ups, because these are quite hard to grab for `helium` since they don’t have any text on their close buttons.

```python
from smolagents improt tool

@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()

@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    # Common selectors for modal close buttons and overlay elements
    modal_selectors = [
        "button[class*='close']",
        "[class*='modal']",
        "[class*='modal'] button",
        "[class*='CloseButton']",
        "[aria-label*='close']",
        ".modal-close",
        ".close-modal",
        ".modal .close",
        ".modal-backdrop",
        ".modal-overlay",
        "[class*='overlay']"
    ]

    wait = WebDriverWait(driver, timeout=0.5)

    for selector in modal_selectors:
        try:
            elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
            )

            for element in elements:
                if element.is_displayed():
                    try:
                        # Try clicking with JavaScript as it's more reliable
                        driver.execute_script("arguments[0].click();", element)
                    except ElementNotInteractableException:
                        # If JavaScript click fails, try regular click
                        element.click()

        except TimeoutException:
            continue
        except Exception as e:
            print(f"Error handling selector {selector}: {str(e)}")
            continue
    return "Modals closed"
```

Now let’s move on to defining our agent. We set the highest `verbosity_level` to display the LLM’s full output messages to view its thoughts, and we increased `max_steps` to 20 to give the agent more steps to explore the web.

```python
agent = CodeAgent(
    tools=[go_back, close_popups],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks = [save_screenshot],
    max_steps=20,
    verbosity_level=2
)
```

Finally, we provide our agent with some guidance about using helium.

```python
helium_instructions = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
First you need to import everything from helium, then you can do other actions!
Code:
```py
from helium import *
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scrol_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=100000) # This will probably scroll all the way down
```<end_code>
When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To find elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually!
Of course you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url. Don't kill the browser.
"""
```

And finally, let’s run it!

```python
agent.run("""
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
""" + helium_instructions)
```

## Next Steps

This will give you the glimpse of the capabilities of a vision-enabled `CodeAgent`, but there’s much more to do! Get started with the example below.

- You can get started with the agentic web browser [here](https://github.com/huggingface/smolagents/blob/main/examples/vlm_web_browser.py).
- Read more about smolagents [in our announcement blog post](https://huggingface.co/blog/smolagents).
- Read [the smolagents documentation](https://huggingface.co/docs/smolagents/index).

We are looking forward to see what you will build with vision language models and smolagents!
