// Simple script to check if OpenAI API key is working
require('dotenv').config({ path: '.env.local' });
const { OpenAI } = require('openai');

async function checkOpenAIKey() {
  console.log('Checking OpenAI API key...');
  
  // Check if key exists
  if (!process.env.OPENAI_API_KEY) {
    console.error('Error: OPENAI_API_KEY is not set in .env.local file');
    return;
  }
  
  console.log('API key found in environment variables');
  console.log('Key starts with:', process.env.OPENAI_API_KEY.substring(0, 3) + '...');
  
  try {
    // Initialize OpenAI client
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
    
    // Try a simple completion to test the key
    console.log('Testing API key with a simple request...');
    const response = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: 'Hello, are you working?' }],
      max_tokens: 10
    });
    
    console.log('API key is working! Response:', response.choices[0]?.message?.content);
  } catch (error) {
    console.error('Error testing OpenAI API key:', error.message);
    if (error.message.includes('API key')) {
      console.error('This appears to be an API key issue. Please check that your key is valid and has not expired.');
    } else if (error.message.includes('rate limit')) {
      console.error('You have hit the rate limit. Please try again later or upgrade your OpenAI plan.');
    }
  }
}

checkOpenAIKey();
