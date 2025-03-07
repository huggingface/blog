---
title: "LLM Inference on Edge: A Fun and Easy Guide to run LLMs via React Native on your Phone!"
thumbnail: /blog/assets/llm_inference_on_edge/thumbnail.png
authors:
  - user: medmekk
  - user: marcsun13
---

As LLMs continue to evolve, they are becoming smaller and smarter, enabling them to run directly on your phone. Take, for instance, the DeepSeek R1 Distil Qwen 2.5 with 1.5 billion parameters, this model really shows how advanced AI can now fit into the palm of your hand!

In this blog, we will guide you through creating a mobile app that allows you to chat with these powerful models locally. The complete code for this tutorial is available in our [EdgeLLM repository](https://github.com/MekkCyber/EdgeLLM). If you've ever felt overwhelmed by the complexity of open-source projects, fear not! Inspired by the [Pocket Pal](https://github.com/a-ghorbani/pocketpal-ai) app, we will help you build a straightforward React Native application that downloads LLMs from the [**Hugging Face**](https://huggingface.co/) hub, ensuring everything remains private and runs on your device. We will utilize `llama.rn`, a binding for `llama.cpp`, to load GGUF files efficiently!

## Why You Should Follow This Tutorial?

This tutorial is designed for anyone who:

- Is interested in integrating AI into mobile applications
- Wants to create a conversational app compatible with both Android and iOS using React Native
- Seeks to develop privacy-focused AI applications that operate entirely offline

By the end of this guide, you will have a fully functional app that allows you to interact with your favorite models.

<div style="display: flex; justify-content: center;">
  <video src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/EdgeLLM_demo.mp4" width="960" height="720" controls ></video>
</div>

---
## 0. Choosing the Right Models

Before we dive into building our app, let's talk about which models work well on mobile devices and what to consider when selecting them.

### Model Size Considerations

When running LLMs on mobile devices, size matters significantly:

- **Small models (1-3B parameters)**: Ideal for most mobile devices, offering good performance with minimal latency
- **Medium models (4-7B parameters)**: Work well on newer high-end devices but may cause slowdowns on older phones
- **Large models (8B+ parameters)**: Generally too resource-intensive for most mobile devices, but can be used if quantized to low precision formats like Q2_K or Q4_K_M

### GGUF Quantization Formats

When downloading GGUF models, you'll encounter various quantization formats. Understanding these can help you choose the right balance between model size and performance:

#### Legacy Quants (Q4_0, Q4_1, Q8_0)
- Basic, straightforward quantization methods
- Each block is stored with:  
	•	Quantized values (the compressed weights).  
	•	One (_0) or two (_1) scaling constants.  
- Fast but less efficient than newer methods => not used widely anymore

#### K-Quants (Q3_K_S, Q5_K_M, ...)
- Introduced in this [PR](https://github.com/ggml-org/llama.cpp/pull/1684)
- Smarter bit allocation than legacy quants
- The K in “K-quants” refers to a mixed quantization format, meaning some layers get more bits for better accuracy.
- Suffixes like _XS, _S, or _M refer to specific mixes of quantization (smaller = more compression), for example :  
  • Q3_K_S uses Q3_K for all tensors  
  •	Q3_K_M uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, and Q3_K for the rest.  
  •	Q3_K_L uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, and Q5_K for the rest.  

#### I-Quants (IQ2_XXS, IQ3_S, ...)
- It still uses the block-based quantization, but with some new features inspired by QuIP
- Smaller file sizes but may be slower on some hardware
- Best for devices with strong compute power but limited memory

### Recommended Models to Try

Here are some models that perform well on mobile devices:

1. **[SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)**
2. **[Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)**
3. **[Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)**
4. **[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)**

### Finding More Models

To find additional GGUF models on Hugging Face:

1. Visit [huggingface.co/models](https://huggingface.co/models)
2. Use the search filters:
   - Visit the [GGUF models page](https://huggingface.co/models?library=gguf)
   - Specify the size of the model in the search bar
   - Look for "chat" or "instruct" in the name for conversational models

![Hugging Face search filters](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/find_gguf_models.png)

When selecting a model, consider both the parameter count and the quantization level. For example, a 7B model with Q2_K quantization might run better than a 2B model with Q8_0 quantization. So if you can fit a small model comfortably on your device try to use a bigger quantized model instead, it might have a better performance.

## 1. Setting Up Your Environment

React Native is a popular framework for building mobile applications using JavaScript and React. It allows developers to create apps that run on both Android and iOS platforms while sharing a significant amount of code, which speeds up the development process and reduces maintenance efforts.

Before you can start coding with React Native, you need to set up your environment properly.

### Tools You Need

1. **Node.js:** Node.js is a JavaScript runtime that allows you to run JavaScript code. It is essential for managing packages and dependencies in your React Native project. You can install it from [Node.js downloads](https://nodejs.org/en/download).

2. **react-native-community/cli:** This command installs the React Native command line interface (CLI), which provides tools to create, build, and manage your React Native projects. Run the following command to install it:

```bash
npm i @react-native-community/cli
```


### Virtual Device Setup

To run your app during development, you will need an emulator or a simulator:

- **If you are on macOS:**
  - For iOS: Install Xcode -> Open Developer Tools -> Simulator
  ![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/xcode_simulator.png)
  - For Android: Install Java Runtime and Android Studio -> Go to Device Manager and Create an emulator
  ![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/android.png)

- **If you are on Windows or Linux:**
  - For iOS: We need to rely on cloud-based simulators like [LambdaTest](https://www.lambdatest.com/test-on-iphone-simulator) and [BrowserStack](https://www.browserstack.com/test-on-ios-simulator)
  - For Android: Install Java Runtime and Android Studio -> Go to Device Manager and Create an emulator

If you are curious about the difference between simulators and emulators, you can read this article: [Difference between Emulator and Simulator](https://www.browserstack.com/guide/difference-between-emulator-and-simulator), but to put it simply, emulators replicate both hardware and software, while simulators only replicate software.

For setting up Android Studio, follow this excellent tutorial by Expo : [Android Studio Emulator Guide](https://docs.expo.dev/workflow/android-studio-emulator/)

## 2. Create the App

Let's start this project! 

You can find the full code for this project in the `EdgeLLM` repo [here](https://github.com/MekkCyber/EdgeLLM), there are two folders:

- `EdgeLLMBasic`: A basic implementation of the app with a simple chat interface
- `EdgeLLMPlus`: An enhanced version of the app with a more complex chat interface and additional features

First, we need to initiate the app using @react-native-community/cli:

```bash
npx @react-native-community/cli@latest init <ProjectName>
```

### Project Structure 

App folders are organized as follows:

#### Default Files/Folders

1. `android/`

   - Contains native Android project files
   - **Purpose**: To build and run the app on Android devices

2. `ios/`

   - Contains native iOS project files
   - **Purpose**: To build and run the app on iOS devices

3. `node_modules/`

   - **Purpose**: Holds all npm dependencies used in the project

4. `App.tsx`

   - The main root component of your app, written in TypeScript
   - **Purpose**: Entry point to the app's UI and logic

5. `index.js`
   - Registers the root component (`App`)
   - **Purpose**: Entry point for the React Native runtime. You don't need to modify this file.

<details>
<summary><b>Additional Configuration Files</b></summary>

- `tsconfig.json`: Configures TypeScript settings
- `babel.config.js`: Configures Babel for transpiling modern JavaScript/TypeScript, which means it will convert modern JS/TS code to older JS/TS code that is compatible with older browsers or devices.
- `jest.config.js`: Configures Jest for testing React Native components and logic.
- `metro.config.js`: Customizes the Metro bundler for the project. It's a JavaScript bundler specifically designed for React Native. It takes your project's JavaScript and assets, bundles them into a single file (or multiple files for efficient loading), and serves them to the app during development. Metro is optimized for fast incremental builds, supports hot reloading, and handles React Native's platform-specific files (.ios.js or .android.js).
- `.watchmanconfig`: Configures Watchman, a file-watching service used by React Native for hot reloading.
</details>

## 3. Running the Demo & Project

### Running the Demo
To run the project, and see how it looks like on your own virtual device, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MekkCyber/EdgeLLM.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd EdgeLLMPlus 
   #or 
   cd EdgeLLMBasic
   ```

3. **Install Dependencies**:
   ```bash
   npm install
   ```

4. **Navigate to the iOS Folder and Install**:
   ```bash
   cd ios
   pod install
   ```

5. **Start the Metro Bundler**:
   Run the following command in the project folder (EdgeLLMPlus or EdgeLLMBasic):
   ```bash
   npm start
   ```

6. **Launch the App on iOS or Android Simulator**:
   Open another terminal and run:
   ```bash
   # For iOS
   npm run ios

   # For Android
   npm run android
   ```

This will build and launch the app on your emulator/simulator to test the project before we start coding.

### Running the Project

Running a React Native application requires either an emulator/simulator or a physical device. We'll focus on using an emulator since it provides a more streamlined development experience with your code editor and debugging tools side by side.

We start by ensuring our development environment is ready, we need to be in the project folder and run the following commands:

```bash
# Install dependencies
npm install

# Start the Metro bundler
npm start
```

In a new terminal, we will launch the app on our chosen platform:

```bash
# For iOS
npm run ios

# For Android
npm run android
```

This should build and launch the app on your emulator/simulator.

## 4. App Implementation  

### Installing Dependencies

First, let's install the required packages. We aim to load models from the [Hugging Face Hub](https://huggingface.co/) and run them locally. To achieve this, we need to install :

- [`llama.rn`](https://github.com/mybigday/llama.rn): a binding for [`llama.cpp`](https://github.com/ggerganov/llama.cpp) for React Native apps.
- `react-native-fs`: allows us to manage the device's file system in a React Native environment.
- `axios`: a library for sending requests to the Hugging Face Hub API.

```bash
npm install axios react-native-fs llama.rn
```

Let's run the app on our emulator/simulator as we showed before so we can start the development

### State Management

We will start by deleting everyting from the `App.tsx` file, and creating an empty code structure like the following :

<details>
<summary><b>App.tsx</b></summary>

```typescript
import React from 'react';
import {StyleSheet, Text, View} from 'react-native';

function App(): React.JSX.Element {
  return <View> <Text>Hello World</Text> </View>;
}
const styles = StyleSheet.create({});

export default App;
```

</details>
<br>

Inside the `return` statement of the `App` function we define the UI rendered, and outside we define the logic, but all code will be inside the `App` function.

We will have a screen that looks like this:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/hello_world.png" alt="Hello World app" width="300" />
</div>

The text "Hello World" is not displayed properly because we are using a simple `View` component, we need to use a `SafeAreaView` component to display the text correctly, we will deal with that in the next sections.

Now let's think about what our app needs to track for now:

1. **Chat-related**:

   - The conversation history (messages between user and AI)
   - Current user input

2. **Model-related**:
   - Selected model format (like Llama 1B or Qwen 1.5B)
   - Available GGUF files list for each model format
   - Selected GGUF file to download
   - Model download progress
   - A context to store the loaded model
   - A boolean to check if the model is downloading
   - A boolean to check if the model is generating a response

Here's how we implement these states using React's useState hook (we will need to import it from react)

<details>
<summary><b>State Management Code</b></summary>

```typescript
import { useState } from 'react';
...
type Message = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

const INITIAL_CONVERSATION: Message[] = [
    {
      role: 'system',
      content:
        'This is a conversation between user and assistant, a friendly chatbot.',
    },
];

const [conversation, setConversation] = useState<Message[]>(INITIAL_CONVERSATION);
const [selectedModelFormat, setSelectedModelFormat] = useState<string>('');
const [selectedGGUF, setSelectedGGUF] = useState<string | null>(null);
const [availableGGUFs, setAvailableGGUFs] = useState<string[]>([]);
const [userInput, setUserInput] = useState<string>('');
const [progress, setProgress] = useState<number>(0);
const [context, setContext] = useState<any>(null);
const [isDownloading, setIsDownloading] = useState<boolean>(false);
const [isGenerating, setIsGenerating] = useState<boolean>(false);
```

</details>

<br>

This will be added to the `App.tsx` file inside the `App` function but outside the `return` statement as it's part of the logic.

The Message type defines the structure of chat messages, specifying that each message must have a role (either 'user' or 'assistant' or 'system') and content (the actual message text).

Now that we have our basic state management set up, we need to think about how to:

1. Fetch available GGUF models from [Hugging Face](https://huggingface.co/)
2. Download and manage models locally
3. Create the chat interface
4. Handle message generation

Let's tackle these one by one in the next sections...

### Fetching available GGUF models from the Hub

Let's start by defining the model formats our app is going to support and their repositories. Of course `llama.rn` is a binding for `llama.cpp` so we need to load `GGUF` files. To find GGUF repositories for the models we want to support, we can use the search bar on [Hugging Face](https://huggingface.co/) and search for `GGUF` files for a specific model, or use the script `quantize_gguf.py` provided [here](https://github.com/MekkCyber/EdgeLLM/blob/main/quantize_gguf.py) to quantize the model ourselves and upload the files to our hub repository.

```typescript
const modelFormats = [
  {label: 'Llama-3.2-1B-Instruct'},
  {label: 'Qwen2-0.5B-Instruct'},
  {label: 'DeepSeek-R1-Distill-Qwen-1.5B'},
  {label: 'SmolLM2-1.7B-Instruct'},
];

const HF_TO_GGUF = {
    "Llama-3.2-1B-Instruct": "medmekk/Llama-3.2-1B-Instruct.GGUF",
    "DeepSeek-R1-Distill-Qwen-1.5B":
      "medmekk/DeepSeek-R1-Distill-Qwen-1.5B.GGUF",
    "Qwen2-0.5B-Instruct": "medmekk/Qwen2.5-0.5B-Instruct.GGUF",
    "SmolLM2-1.7B-Instruct": "medmekk/SmolLM2-1.7B-Instruct.GGUF",
  };
```

The `HF_TO_GGUF` object maps user-friendly model names to their corresponding Hugging Face repository paths. For example:

- When a user selects 'Llama-3.2-1B-Instruct', it maps to [`medmekk/Llama-3.2-1B-Instruct.GGUF`](https://huggingface.co/medmekk/Llama-3.2-1B-Instruct-GGUF) which is one of the repositories containing the GGUF files for the Llama 3.2 1B Instruct model.

The `modelFormats` array contains the list of model options that will be displayed to users in the selection screen, we chose [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B), [DeepSeek R1 Distill Qwen 1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [Qwen 2 0.5B Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) and [SmolLM2 1.7B Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) as they are the most popular small models.

Next, let's create a way to fetch and display available GGUF model files from the hub for our selected model format.

When a user selects a model format, we make an API call to Hugging Face using the repository path we mapped in our `HF_TO_GGUF` object. We're specifically looking for files that end with '.gguf' extension, which are our quantized model files.

Once we receive the response, we extract just the filenames of these GGUF files and store them in our `availableGGUFs` state using `setAvailableGGUFs`. This allows us to show users a list of available GGUF model variants they can download.

<details>
<summary><b>Fetching Available GGUF Files</b></summary>

```typescript
const fetchAvailableGGUFs = async (modelFormat: string) => {
  if (!modelFormat) {
    Alert.alert('Error', 'Please select a model format first.');
    return;
  }

  try {
    const repoPath = HF_TO_GGUF[modelFormat as keyof typeof HF_TO_GGUF];
    if (!repoPath) {
      throw new Error(
        `No repository mapping found for model format: ${modelFormat}`,
      );
    }

    const response = await axios.get(
      `https://huggingface.co/api/models/${repoPath}`,
    );

    if (!response.data?.siblings) {
      throw new Error('Invalid API response format');
    }

    const files = response.data.siblings.filter((file: {rfilename: string}) =>
      file.rfilename.endsWith('.gguf'),
    );

    setAvailableGGUFs(files.map((file: {rfilename: string}) => file.rfilename));
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : 'Failed to fetch .gguf files';
    Alert.alert('Error', errorMessage);
    setAvailableGGUFs([]);
  }
};
```

</details>
<br>

> **Note:** Ensure to import axios and Alert at the top of your file if not already imported.

We need to test that the function is working correclty, let's add a button to the UI to trigger the function, instead of `View` we will use a `SafeAreaView` (more on that later) component, and we will display the available GGUF files in a `ScrollView` component. the `onPress` function is triggered when the button is pressed:

```typescript
<TouchableOpacity onPress={() => fetchAvailableGGUFs('Llama-3.2-1B-Instruct')}>
  <Text>Fetch GGUF Files</Text>
</TouchableOpacity>
<ScrollView>
  {availableGGUFs.map((file) => (
    <Text key={file}>{file}</Text>
  ))}
</ScrollView>
```
This should look something like this : 

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/available_gguf_files_test.png" alt="Available GGUF Files" width="300" />
</div>

> **Note:** For the whole code until now you can check the `first_checkpoint` branch in the `EdgeLLMBasic` folder [here](https://github.com/MekkCyber/EdgeLLM/blob/first_checkpoint/EdgeLLMBasic/App.tsx)

### Model Download Implementation

Now let's implement the model download functionality in the `handleDownloadModel` function which should be called when the user clicks on the download button. This will download the selected GGUF file from Hugging Face and store it in the app's Documents directory:

<details>
<summary><b>Model Download Function</b></summary>

```typescript
const handleDownloadModel = async (file: string) => {
  const downloadUrl = `https://huggingface.co/${
    HF_TO_GGUF[selectedModelFormat as keyof typeof HF_TO_GGUF]
  }/resolve/main/${file}`;
  // we set the isDownloading state to true to show the progress bar and set the progress to 0
  setIsDownloading(true);
  setProgress(0);

  try {
    // we download the model using the downloadModel function, it takes the selected GGUF file, the download URL, and a progress callback function to update the progress bar
    const destPath = await downloadModel(file, downloadUrl, progress =>
      setProgress(progress),
    );
  } catch (error) {
    const errorMessage =
      error instanceof Error
        ? error.message
        : 'Download failed due to an unknown error.';
    Alert.alert('Error', errorMessage);
  } finally {
    setIsDownloading(false);
  }
};
```

</details>
<br>

We could have implemented the `api` requests inside the `handleDownloadModel` function, but we will keep it in a separate file to keep the code clean and readable. `handleDownloadModel` calls the `downloadModel` function, located in `src/api`, which accepts three parameters: `modelName`, `downloadUrl`, and a `progress` callback function. This callback is triggered during the download process to update the progress. Before downloading we need to have the `selectedModelFormat` state set to the model format we want to download.

Inside the `downloadModel` function we use the `RNFS` module, part of the `react-native-fs` library, to access the device's file system. It allows developers to read, write, and manage files on the device's storage. In this case, the model is stored in the app's Documents folder using `RNFS.DocumentDirectoryPath`, ensuring that the downloaded file is accessible to the app. The progress bar is updated accordingly to reflect the current download status and the progress bar component is defined in the `components` folder.

Let's create `src/api/model.ts` and copy the code from the [`src/api/model.ts`](https://github.com/MekkCyber/EdgeLLM/blob/main/EdgeLLMBasic/src/api/model.ts) file in the repo. The logic should be simple to understand. The same goes for the progress bar component in the [`src/components`](https://github.com/MekkCyber/EdgeLLM/blob/main/EdgeLLMBasic/src/components/ProgressBar.tsx) folder, it's a simple colored `View` where the width is the progress of the download.

Now we need to test the `handleDownloadModel` function, let's add a button to the UI to trigger the function, and we will display the progress bar. This will be added under the `ScrollView` we added before.

<details>
<summary><b>Download Model Button</b></summary>

```typescript
<View style={{ marginTop: 30, marginBottom: 15 }}>
  {Object.keys(HF_TO_GGUF).map((format) => (
    <TouchableOpacity
      key={format}
      onPress={() => {
        setSelectedModelFormat(format);
      }}
    >
      <Text> {format} </Text>
    </TouchableOpacity>
  ))}
</View>
<Text style={{ marginBottom: 10, color: selectedModelFormat ? 'black' : 'gray' }}>
  {selectedModelFormat 
    ? `Selected: ${selectedModelFormat}` 
    : 'Please select a model format before downloading'}
</Text>
<TouchableOpacity
  onPress={() => {
    handleDownloadModel("Llama-3.2-1B-Instruct-Q2_K.gguf");
  }}
>
  <Text>Download Model</Text>
</TouchableOpacity>
{isDownloading && <ProgressBar progress={progress} />}
```

</details>
<br>

In the UI we show a list of the supported model formats and a button to download the model, when the user chooses the model format and clicks on the button the progress bar should be displayed and the download should start. In the test we hardcoded the model to download `Llama-3.2-1B-Instruct-Q2_K.gguf`, so we need to select `Llama-3.2-1B-Instruct` as a model format for the function to work, we should have something like:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/download_image.png" alt="Download Model" width="300" />
</div>

> **Note:** For the whole code until now you can check the `second_checkpoint` branch in the `EdgeLLMBasic` folder [here](https://github.com/MekkCyber/EdgeLLM/blob/second_checkpoint/EdgeLLMBasic/App.tsx)


### Model Loading and Initialization

Next, we will implement a function to load the downloaded model into a Llama context, as detailed in the `llama.rn` documentation available [here](https://github.com/mybigday/llama.rn). If a context is already present, we will release it, set the context to `null`, and reset the conversation to its initial state. Subsequently, we will utilize the `initLlama` function to load the model into a new context and update our state with the newly initialized context.

<details>
<summary><b>Model Loading Function</b></summary>

```typescript
import {initLlama, releaseAllLlama} from 'llama.rn';
import RNFS from 'react-native-fs'; // File system module
...
const loadModel = async (modelName: string) => {
  try {
    const destPath = `${RNFS.DocumentDirectoryPath}/${modelName}`;

    // Ensure the model file exists before attempting to load it
    const fileExists = await RNFS.exists(destPath);
    if (!fileExists) {
      Alert.alert('Error Loading Model', 'The model file does not exist.');
      return false;
    }

    if (context) {
      await releaseAllLlama();
      setContext(null);
      setConversation(INITIAL_CONVERSATION);
    }

    const llamaContext = await initLlama({
      model: destPath,
      use_mlock: true,
      n_ctx: 2048,
      n_gpu_layers: 1
    });
    console.log("llamaContext", llamaContext);
    setContext(llamaContext);
    return true;
  } catch (error) {
    Alert.alert('Error Loading Model', error instanceof Error ? error.message : 'An unknown error occurred.');
    return false;
  }
};
```

</details>
<br>

We need to call the `loadModel` function when the user clicks on the download button, so we need to add it inside the `handleDownloadModel` function right after the download is complete if it's successful.

```typescript
// inside the handleDownloadModel function, just after the download is complete 
if (destPath) {
  await loadModel(file);
}
```
To test the model loading let's add a `console.log` inside the `loadModel` function to print the context, so we can see if the model is loaded correctly. We keep the UI the same as before, because clicking on the download button will trigger the `handleDownloadModel` function, and the `loadModel` function will be called inside it. To see the `console.log` output we need to open the Developer Tools, for that we press `j` in the terminal where we ran `npm start`. If everything is working correctly we should see the context printed in the console.
![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/llama_context.png)

> **Note:** For the whole code until now you can check the `third_checkpoint` branch in the `EdgeLLMBasic` folder [here](https://github.com/MekkCyber/EdgeLLM/blob/third_checkpoint/EdgeLLMBasic/App.tsx)

### Chat Implementation

With the model now loaded into our context, we can proceed to implement the conversation logic. We'll define a function called `handleSendMessage`, which will be triggered when the user submits their input. This function will update the conversation state and send the updated conversation to the model via `context.completion`. The response from the model will then be used to further update the conversation, which means that the conversation will be updated twice in this function.

<details>
<summary><b>Chat Function</b></summary>

```typescript
const handleSendMessage = async () => {
  // Check if context is loaded and user input is valid
  if (!context) {
    Alert.alert('Model Not Loaded', 'Please load the model first.');
    return;
  }

  if (!userInput.trim()) {
    Alert.alert('Input Error', 'Please enter a message.');
    return;
  }

  const newConversation: Message[] = [
    // ... is a spread operator that spreads the previous conversation array to which we add the new user message
    ...conversation,
    {role: 'user', content: userInput},
  ];
  setIsGenerating(true);
  // Update conversation state and clear user input
  setConversation(newConversation);
  setUserInput('');

  try {
    // we define list the stop words for all the model formats
    const stopWords = [
      '</s>',
      '<|end|>',
      'user:',
      'assistant:',
      '<|im_end|>',
      '<|eot_id|>',
      '<|end▁of▁sentence|>',
      '<｜end▁of▁sentence｜>',
    ];
    // now that we have the new conversation with the user message, we can send it to the model
    const result = await context.completion({
      messages: newConversation,
      n_predict: 10000,
      stop: stopWords,
    });

    // Ensure the result has text before updating the conversation
    if (result && result.text) {
      setConversation(prev => [
        ...prev,
        {role: 'assistant', content: result.text.trim()},
      ]);
    } else {
      throw new Error('No response from the model.');
    }
  } catch (error) {
    // Handle errors during inference
    Alert.alert(
      'Error During Inference',
      error instanceof Error ? error.message : 'An unknown error occurred.',
    );
  } finally {
    setIsGenerating(false);
  }
};
```

</details>
<br>

To test the `handleSendMessage` function we need to add an input text field and a button to the UI to trigger the function, and we will display the conversation in the `ScrollView` component.

<details>
<summary><b>Simple Chat UI</b></summary>

```typescript
<View
  style={{
    flexDirection: "row",
    alignItems: "center",
    marginVertical: 10,
    marginHorizontal: 10,
  }}
>
  <TextInput
    style={{flex: 1, borderWidth: 1}}
    value={userInput}
    onChangeText={setUserInput}
    placeholder="Type your message here..."
  />
  <TouchableOpacity
    onPress={handleSendMessage}
    style={{backgroundColor: "#007AFF"}}
  >
    <Text style={{ color: "white" }}>Send</Text>
  </TouchableOpacity>
</View>
<ScrollView>
  {conversation.map((msg, index) => (
    <Text style={{marginVertical: 10}} key={index}>{msg.content}</Text>
  ))}
</ScrollView>
```

</details>
<br>

If everything is implemented correctly, we should be able to send messages to the model and see the conversation in the `ScrollView` component, it's not beautiful of course but it's a good start, we will improve the UI later.
The result should look like this:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/chat.png" alt="Chat" width="300" />
</div>

> **Note:** For the whole code until now you can check the `fourth_checkpoint` branch in the `EdgeLLMBasic` folder [here](https://github.com/MekkCyber/EdgeLLM/blob/fourth_checkpoint/EdgeLLMBasic/App.tsx)

### The UI & Logic

Now that we have the core functionality implemented, we can focus on the UI. The UI is straightforward, consisting of a model selection screen with a list of models and a chat interface that includes a conversation history and a user input field. During the model download phase, a progress bar is displayed. We intentionally avoid adding many screens to keep the app simple and focused on its core functionality. To keep track of which part of the app is being used, we will use a an other state variable called `currentPage`, it will be a string that can be either `modelSelection` or `conversation`. We add it to the `App.tsx` file.

```typescript
const [currentPage, setCurrentPage] = useState<
  'modelSelection' | 'conversation'
>('modelSelection'); // Navigation state
```
For the css we will use the same styles as in the [EdgeLLMBasic](https://github.com/MekkCyber/EdgeLLM/blob/main/EdgeLLMBasic/App.tsx#L370) repo, you can copy the styles from there.

We will start by working on the model selection screen in the App.tsx file, we will add a list of model formats (you need to do the necessary imports and delete the previous code in the `SafeAreaView` component we used for testing):

<details>
<summary><b>Model Selection UI</b></summary>

```typescript
<SafeAreaView style={styles.container}>
  <ScrollView contentContainerStyle={styles.scrollView}>
    <Text style={styles.title}>Llama Chat</Text>
    {/* Model Selection Section */}
      {currentPage === 'modelSelection' && (
        <View style={styles.card}>
          <Text style={styles.subtitle}>Choose a model format</Text>
          {modelFormats.map(format => (
            <TouchableOpacity
              key={format.label}
              style={[
                styles.button,
                selectedModelFormat === format.label && styles.selectedButton,
              ]}
              onPress={() => handleFormatSelection(format.label)}>
              <Text style={styles.buttonText}>{format.label}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}
  </ScrollView>
</SafeAreaView>
```

</details>
<br>

We use `SafeAreaView` to ensure that the app is displayed correctly on devices with different screen sizes and orientations as we did in the previous section, and we use `ScrollView` to allow the user to scroll through the model formats. We also use `modelFormats.map` to map over the `modelFormats` array and display each model format as a button with a style that changes when the model format is selected. We also use the `currentPage` state to display the model selection screen only when the `currentPage` state is set to `modelSelection`, this is done by using the `&&` operator. The `TouchableOpacity` component is used to allow the user to select a model format by pressing on it.

Now let's define `handleFormatSelection` in the App.tsx file:

```typescript
const handleFormatSelection = (format: string) => {
  setSelectedModelFormat(format);
  setAvailableGGUFs([]); // Clear any previous list
  fetchAvailableGGUFs(format);
};
```

We store the selected model format in the state and clear the previous list of GGUF files from other selections, and then we fetch the new list of GGUF files for the selected format.
The screen should look like this on your device:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/model_selection_start.png" alt="Model Selection Start" width="300" />
</div>

Next, let's add the view to show the list of GGUF files already available for the selected model format, we will add it below the model format selection section.

<details>
<summary><b>Available GGUF Files UI</b></summary>

```typescript
{
  selectedModelFormat && (
    <View>
      <Text style={styles.subtitle}>Select a .gguf file</Text>
      {availableGGUFs.map((file, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.button,
            selectedGGUF === file && styles.selectedButton,
          ]}
          onPress={() => handleGGUFSelection(file)}>
          <Text style={styles.buttonTextGGUF}>{file}</Text>
        </TouchableOpacity>
      ))}
    </View>
  )
}
```

</details>
<br>

We need to only show the list of GGUF files if the `selectedModelFormat` state is not null, which means a model format is selected by the user.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/available_ggufs.png" alt="Available GGUF Files" width="300" />
</div>

We need to define `handleGGUFSelection` in the App.tsx file as a function that will trigger an alert to confirm the download of the selected GGUF file. If the user clicks on `Yes`, the download will start, else the selected GGUF file will be cleared.

<details>
<summary><b>Confirm Download Alert</b></summary>

```typescript
const handleGGUFSelection = (file: string) => {
  setSelectedGGUF(file);
  Alert.alert(
    'Confirm Download',
    `Do you want to download ${file}?`,
    [
      {
        text: 'No',
        onPress: () => setSelectedGGUF(null),
        style: 'cancel',
      },
      {text: 'Yes', onPress: () => handleDownloadAndNavigate(file)},
    ],
    {cancelable: false},
  );
};
const handleDownloadAndNavigate = async (file: string) => {
  await handleDownloadModel(file);
  setCurrentPage('conversation'); // Navigate to conversation after download
};
```

</details>
<br>

`handleDownloadAndNavigate` is a simple function that will download the selected GGUF file by calling `handleDownloadModel` (implemented in the previous sections) and navigate to the conversation screen after the download is complete.

Now after clicking on a GGUF file, we should have an alert to confirm or cancel the download :

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/confirm_download.png" alt="Confirm Download" width="300" />
</div>

We can add a simple `ActivityIndicator` to the view to display a loading state when the available GGUF files are being fetched. For that we will need to import `ActivityIndicator` from `react-native` and define `isFetching` as a boolean state variable that will be set to true in the start of the `fetchAvailableGGUFs` function and false when the function is finished as you can see here in the [code](https://github.com/MekkCyber/EdgeLLM/blob/main/EdgeLLMBasic/App.tsx#L199), and add the `ActivityIndicator` to the view just before the `{availableGGUFs.map((file, index) => (...))} ` to display a loading state when the available GGUF files are being fetched.

```typescript
{isFetching && (
  <ActivityIndicator size="small" color="#2563EB" />
)}
```
The app should look like this for a brief moment when the GGUF files are being fetched:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/download_indicator.png" alt="Download Indicator" width="300" />
</div>

Now we should be able to see the different GGUF files available for each model format when we click on it, and we should see the alert when clicking on a GGUF confirming if we want to download the model.
Next we need to add the progress bar to the model selection screen, we can do it by importing the `ProgressBar` component from `src/components/ProgressBar.tsx` in the `App.tsx` file as we did before, and we will add it to the view just after the `{availableGGUFs.map((file, index) => (...))} ` to display the progress bar when the model is being downloaded.

<details>
<summary><b>Download Progress Bar</b></summary>

```typescript
{
  isDownloading && (
    <View style={styles.card}>
      <Text style={styles.subtitle}>Downloading : </Text>
      <Text style={styles.subtitle2}>{selectedGGUF}</Text>
      <ProgressBar progress={progress} />
    </View>
  );
}
```

</details>
<br>

The download progress bar will now be positioned at the bottom of the model selection screen. However, this means that users may need to scroll down to view it. To address this, we will modify the display logic so that the model selection screen is only shown when the `currentPage` state is set to 'modelSelection' and the added condition that there is no ongoing model download.

```typescript
{currentPage === 'modelSelection' && !isDownloading && (
  <View style={styles.card}>
  <Text style={styles.subtitle}>Choose a model format</Text>
...
```
After confirming a model download we should have a screen like this :

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/download_progress_bar.png" alt="Download Progress Bar" width="300" />
</div>

> **Note:** For the whole code until now you can check the `fifth_checkpoint` branch in the `EdgeLLMBasic` folder [here](https://github.com/MekkCyber/EdgeLLM/blob/fifth_checkpoint/EdgeLLMBasic/App.tsx)

Now that we have the model selection screen, we can start working on the conversation screen with the chat interface. This screen will be displayed when `currentPage` is set to `conversation`. We will add a conversation history and a user input field to the screen. The conversation history will be displayed in a scrollable view, and the user input field will be displayed at the bottom of the screen out of the scrollable view to stay visible. Each message will be displayed in a different color depending on the role of the message (user or assistant).

We need to add just under the model selection screen the view for the conversation screen:

<details>
<summary><b>Conversation UI</b></summary>

```typescript
{currentPage == 'conversation' && !isDownloading && (
  <View style={styles.chatContainer}>
    <Text style={styles.greetingText}>
      🦙 Welcome! The Llama is ready to chat. Ask away! 🎉
    </Text>
    {conversation.slice(1).map((msg, index) => (
      <View key={index} style={styles.messageWrapper}>
        <View
          style={[
            styles.messageBubble,
            msg.role === 'user'
              ? styles.userBubble
              : styles.llamaBubble,
          ]}>
          <Text
            style={[
              styles.messageText,
              msg.role === 'user' && styles.userMessageText,
            ]}>
              {msg.content}
          </Text>
        </View>
      </View>
    ))}
  </View>
)}
```

</details>
<br>

We use different styles for the user messages and the model messages, and we use the `conversation.slice(1)` to remove the first message from the conversation, which is the system message.

We can now add the user input field at the bottom of the screen and the send button (they should not be inside the `ScrollView`). As I mentioned before, we will use the `handleSendMessage` function to send the user message to the model and update the conversation state with the model response.

<details>
<summary><b>Send Button & Input Field</b></summary>

```typescript
{currentPage === 'conversation' && (
  <View style={styles.inputContainer}>
    <TextInput
      style={styles.input}
      placeholder="Type your message..."
      placeholderTextColor="#94A3B8"
      value={userInput}
      onChangeText={setUserInput}
    />
    <View style={styles.buttonRow}>
      <TouchableOpacity
        style={styles.sendButton}
        onPress={handleSendMessage}
        disabled={isGenerating}>
        <Text style={styles.buttonText}>
          {isGenerating ? 'Generating...' : 'Send'}
        </Text>
      </TouchableOpacity>
    </View>
  </View>
)}
```

</details>
<br>

When the user clicks on the send button, the `handleSendMessage` function will be called and the `isGenerating` state will be set to true. The send button will then be disabled and the text will change to 'Generating...'. When the model finishes generating the response, the `isGenerating` state will be set to false and the text will change back to 'Send'.

> **Note:** For the whole code until now you can check the `main` branch in the `EdgeLLMBasic` folder [here](https://github.com/MekkCyber/EdgeLLM/blob/main/EdgeLLMBasic/App.tsx)

The conversation page should now look like this:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/whole_basic_app.png" alt="Whole Basic App" width="300" />
</div>

Congratulations you've just built the core functionality of your first AI chatbot, the code is available [here](https://github.com/MekkCyber/EdgeLLM/blob/main/EdgeLLMBasic/App.tsx) ! You can now start adding more features to the app to make it more user friendly and efficient.

### The other Functionnalities

The app is now fully functional, you can download a model, select a GGUF file, and chat with the model, but the user experience is not the best. In the [`EdgeLLMPlus`](https://github.com/MekkCyber/EdgeLLM/tree/main/EdgeLLMPlus) repo, I've added some other features, like on the fly generation, automatic scrolling, the inference speed tracking, the thought process of the model like deepseek-qwen-1.5B,... we will not go into details here as it will make the blog too long, we will go through some of the ideas and how to implement them but the whole code is available in the repo

#### Generation on the fly
The app generates responses incrementally, producing one token at a time rather than delivering the entire response in a single batch. This approach enhances the user experience, allowing users to begin reading the response as it is being formed. We achieve this by utilizing a `callback` function within `context.completion`, which is triggered after each token is generated, enabling us to update the `conversation` state accordingly.

#### Auto Scrolling
Auto Scrolling ensures that the latest messages or tokens are always visible to the user by automatically scrolling the chat view to the bottom as new content is added. To implement that we need we use a reference to the `ScrollView` to allow programatic control over the scroll position, and we use the `scrollToEnd` method to scroll to the bottom of the `ScrollView` when a new message is added to the `conversation` state. We also define a `autoScrollEnabled` state variable that will be set to false when the user scrolls up more than 100px from the bottom of the `ScrollView`.

#### Inference Speed Tracking
Inference Speed Tracking is a feature that tracks the time taken to generate each token and displays under each message generated by the model. This feature is easy to implement because the `CompletionResult` object returned by the `context.completion` function contains a `timings` property which is a dictionary containing many metrics about the inference process. We can use the `predicted_per_second` metric to track the speed of the model.

#### Thought Process
The thought process is a feature that displays the thought process of the model like deepseek-qwen-1.5B. The app identifies special tokens like <think> and </think> to handle the model's internal reasoning or "thoughts." When a <think> token is encountered, the app enters a "thought block" where it accumulates tokens that represent the model's reasoning. Once the closing </think> token is detected, the accumulated thought is extracted and associated with the message, allowing users to toggle the visibility of the model's reasoning. To implement this we need to add a `thought` and `showThought` property to the `Message` type. `message.thought` will store the reasoning of the model and `message.showThought` will be a boolean that will be set to true when the user clicks on the message to toggle the visibility of the thought.

#### Markdown Rendering
The app uses the `react-native-markdown-display` package to render markdown in the conversation. This package allows us to render code in a better format.

#### Model Management
We added a `checkDownloadedModels` function to the `App.tsx` file that will check if the model is already downloaded on the device, if it's not we will download it, if it is we will load it into the context directly, and we added some elements in the UI to show if a model is already downloaded or not.

#### Stop/Back Buttons
We added two important buttons in the UI, the stop button and the back button. The stop button will stop the generation of the response and the back button will navigate to the model selection screen. For that, We added a `handleStopGeneration` function to the `App.tsx` file that will stop the generation of the response by calling `context.stop` and set the `isGenerating` state to false. We also added a `handleBack` function to the `App.tsx` file that will navigate to the model selection screen by setting the `currentPage` state to `modelSelection`.

## 5. How to Debug

### Chrome DevTools Debugging

For debugging we use Chrome DevTools as in web development :

1. Press `j` in the Metro bundler terminal to launch Chrome DevTools
2. Navigate to the "Sources" tab

![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-inferencen-on-edge/dev_tools.png)
3. Find your source files  
4. Set breakpoints by clicking on line numbers  
5. Use debugging controls (top right corner):  
   - Step Over - Execute current line
   - Step Into - Enter function call
   - Step Out - Exit current function
   - Continue - Run until next breakpoint

### Common Debugging Tips

1. **Console Logging**

```javascript
console.log('Debug value:', someValue);
console.warn('Warning message');
console.error('Error details');
```
This will log the output in the console of Chrome DevTools

2. **Metro Bundler Issues**
If you encounter issues with the Metro bundler, you can try clearing the cache first:
```bash
# Clear Metro bundler cache
npm start --reset-cache
```

3. **Build Errors**

```bash
# Clean and rebuild
cd android && ./gradlew clean
cd ios && pod install
```

## 6. Additional Features we can add

To enhance the user experience, we can add some features like:

- **Model Management:**
  - Allow users to delete models from the device
  - Add a feature to delete all downloaded models from the device
  - Add a performance tracking feature to the UI to track memory and cpu usage

- **Model Selection:**
  - Allow users to search for a model
  - Allow users to sort models by name, size, etc.
  - Show the model size in the UI
  - Add support for VLMs

- **Chat Interface:**
  - Display the code in color
  - Math Formatting

I'm sure you can think of some really cool features to add to the app, feel free to implement them and share them with the community 🤗


## 7. Acknowledgments

I would like to thank the following people for reviewing this blog post and providing valuable feedback:

- [Pedro Cuenca](https://huggingface.co/pcuenq)
- [Vaibhav Srivastav](https://huggingface.co/reach-vb)
- [Xuan-Son NGUYEN](https://huggingface.co/ngxson)

Their expertise and suggestions helped improve the quality and accuracy of this guide.

## 8. Conclusion

You now have a working React Native app that can:

- Download models from Hugging Face
- Run inference locally
- Provide a smooth chat experience
- Track model's performance

This implementation serves as a foundation for building more sophisticated AI-powered mobile applications. Remember to consider device capabilities when selecting models and tuning parameters.

Happy coding! 🚀