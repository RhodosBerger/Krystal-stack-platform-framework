import { PersonaProvider } from './context/PersonaContext';
import { AuthProvider } from './context/AuthContext';
import MultisiteRouter from './components/MultisiteRouter';

function App() {
  return (
    <AuthProvider>
      <PersonaProvider>
        <div className="min-h-screen bg-industrial-bg text-white selection:bg-neuro-pulse selection:text-industrial-bg overflow-x-hidden">
          <MultisiteRouter />
        </div>
      </PersonaProvider>
    </AuthProvider>
  );
}

export default App;
