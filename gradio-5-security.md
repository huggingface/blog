---
title: "A Security Review of Gradio 5" 
thumbnail: /blog/assets/gradio-5-security/thumbnail.png
authors:
- user: abidlabs
- user: pngwn
---

# A Security Review of Gradio 5

**We audited Gradio 5 so that your machine learning apps are safe!**

In the past few years, [Gradio](https://github.com/gradio-app/gradio/) has become the default way to build machine learning web applications in Python. In just a few lines of code, you can create a user interface for an image generation app, a chatbot, or any other kind of ML app _and_ share it with others using Gradio’s built-in share links or [Hugging Face Spaces](https://huggingface.co/spaces).

```py
import gradio as gr
def generate(seed, prompt):  
    ...  
    return image
    
# gr.Interface creates a web-based UI
gr.Interface(
    generate,   
    inputs=[gr.Slider(), gr.Textbox()],  
    outputs=[gr.Image()]
).launch(share=True)  
# share=True generates a public link instantly
```

Our goal with Gradio is to allow developers to build web applications that work great out-of-the-box for machine learning use cases. This has meant letting you, as a developer, easily build applications that:

*   Scale easily to large numbers of concurrent users
*   Are accessible to as many users as possible
*   Provide consistent UI, UX, and theming
*   Work reliably across a large number of browsers and devices
    
...even if you are not an expert in scaling, accessibility, or UI/UX!

Now, we’re adding **web** **security** to this list. We asked [Trail of Bits](https://www.trailofbits.com/), a well-known cybersecurity company, to conduct an independent audit of Gradio. The security issues they discovered were all fixed ahead of the Gradio 5 release.

This means that machine learning apps that **you build** with Gradio 5 **will follow best practices when it comes to web security** without any significant changes to your code.

## Why a security audit?

In the past couple of years, the Gradio team has worked with the community to patch security vulnerabilities as they are discovered. But as Gradio’s popularity has grown (with >6 million monthly downloads on PyPi and >470,000 Gradio apps on Hugging Face Spaces), ensuring security has become even more important.

So in Gradio 5, we decided to take a different approach -- do a _preemptive_ security audit of the Gradio codebase so that your machine learning applications built with Gradio 5 are safe by default. 

We asked Trail of Bits to conduct an independent and comprehensive audit of Gradio. Their team of experts in AI and Application Security identified security risks in the Gradio codebase in 4 general scenarios:

*   Gradio apps running locally
*   Gradio apps deployed on Hugging Face Spaces or other servers
*   Gradio apps shared with built-in share links 
*   Supply chain vulnerabilities originating from the Gradio CI pipeline

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-5/security-scenarios.png)

Then, we worked closely with Trail of Bits to identify mitigation strategies for each of these risks. Gradio’s simplicity and ease of use, while beneficial for developers, also presented unique security challenges, as we didn’t want developers to need to set up complex security measures like CORS and CSP policies.

By the end of the collaboration, we fixed all of the security risks that were identified by Trail of Bits. All the fixes were validated by Trail of Bits, and are included in the Gradio 5.0 release. While it is impossible to prove the absence of security vulnerabilities, this is a major step in giving reassurance that your Gradio apps are safe.

### **Major findings**

We outline below the major security vulnerabilities that were discovered by Trail of Bits corresponding to the 4 scenarios above. In the interest of transparency and the spirit of open-source, we are making the full security report public, and more details for each of these issues can be found in the report.

**Gradio apps running locally**

*   **TOB-GRADIO-1** and **TOB-GRADIO-2**: Misconfigurations in the server’s CORS policy that, in the context of an authenticated Gradio server, would allow attackers to steal access tokens and take over a victim’s accounts when they visit their malicious website.
    

**Gradio apps deployed on Hugging Face Spaces or other servers**

*   **TOB-GRADIO-3**: A full read GET-based SSRF that would allow attackers to make requests to and read the responses from arbitrary endpoints, including those on the user’s internal network. 
    
*   **TOB-GRADIO-10**: Arbitrary file type uploads that would allow an attacker to host XSS payloads on a user’s Gradio server. In the context of an authenticated Gradio server, an attacker could use this to take over user accounts when the victim accesses an attacker’s malicious website.
    
*   **TOB-GRADIO-13**: A race condition that allows an attacker to reroute user traffic to their server and steal uploaded files or chatbot conversations.
    
*   **TOB-GRADIO-16**: Several components’ post-process functions could allow attackers to leak arbitrary files in very simple Gradio server configurations.
    

**Gradio apps shared with built-in share links**

*   **TOB-GRADIO-19**: Remote code execution (RCE) with the root user on the Gradio API Server via a nginx misconfiguration that exposed the unauthenticated docker API. This allowed an attacker to provide a malicious host and port in step 2 of the diagram and redirect all frp tunnels to a malicious server that records all user traffic, including uploaded files and chatbox conversations.
    
*   **TOB-GRADIO-11**: Lack of robust encryption in communications between the frp-client and frp-server, allowing attackers in a position to intercept requests (the ones from steps 6 and 7 in the diagram above) to read and modify the data going to and from the frp-server.
    

**Supply chain vulnerabilities originating from the Gradio CI pipeline**

*   **TOB-GRADIO-25**: Several GitHub Actions workflows in the Gradio repository use third-party actions pinned to tags or branch names instead of full commit SHAs. This could allow malicious actors to silently modify actions, potentially leading to tampering with application releases or leaking secrets.
    
*   Separately, a [GitHub security researcher reported](https://github.com/gradio-app/gradio/security/advisories/GHSA-48pj-2428-pp3w) that our GitHub actions could allow untrusted code execution and secret exfiltration if an attacker triggered a workflow and cleverly dumped the memory of GitHub runners. 
    

### **Going forward**

We are committed to ensuring that as we continue to develop Gradio 5 (and we have lots of plans!), we do so in a manner that prioritizes security so that we can do our part in making machine learning applications better and safer! 

In addition to continuing to work with the security community, we have also added security unit tests to our test suite, fuzzer tests specifically designed to identify potential vulnerabilities, and are using static analysis tools like Semgrep in our CI to detect common security issues in our code and prevent security regressions.

We’re very grateful to Trail of Bits for the comprehensive security audit of Gradio and for validating the mitigations that we put in place for Gradio 5.