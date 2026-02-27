import { BrowserRouter, Route, Routes } from 'react-router-dom'
import HomePage from './pages/home'

function App() {
  return (
    <BrowserRouter basename="/simulator">
      <Routes>
        <Route path="/" element={<HomePage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
