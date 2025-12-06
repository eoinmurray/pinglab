import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ThemeProvider } from "./components/ThemeProvider"
import { Page } from "./pages/Page"
import { Slides } from "./pages/Slides"
import Ministry from "./pages/Ministry"

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <BrowserRouter>
          <Routes>
            <Route path="/ministry" element={<Ministry />} />
            <Route path=":path/SLIDES.mdx" element={<Slides />} />
            <Route path="/*" element={<Page />} />
          </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
