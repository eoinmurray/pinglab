import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ThemeProvider } from "./components/ThemeProvider"
import { Page } from "./pages/Page"
import { Slides } from "./pages/Slides"

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <BrowserRouter>
          <Routes>
            <Route path=":path/SLIDES.mdx" element={<Slides />} />
            <Route path="/*" element={<Page />} />
          </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
