import ollama from 'ollama';
import readline from 'readline';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.question('Ask something: ', async (question) => {
  const message = { role: 'user', content: question };
  const response = await ollama.chat({ model: 'llama2-uncensored', messages: [message], stream: true });
  for await (const part of response) {
    process.stdout.write(part.message.content);
  }
  
  rl.close();
});
