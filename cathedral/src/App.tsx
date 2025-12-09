import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ThemeProvider } from "./components/ThemeProvider"
import { Home } from "./pages/Home"
import { Post } from "./pages/Post"
import { SlidesPage } from "./pages/Slides"

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <BrowserRouter>
          <Routes>
            <Route path=":path/SLIDES.mdx" element={<SlidesPage />} />
            <Route path=":path/README.mdx" element={<Post />} />
            <Route path="/*" element={<Home />} />
          </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
