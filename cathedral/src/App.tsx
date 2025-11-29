import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ThemeProvider } from "./components/ThemeProvider"
import { Page } from "./pages/Page"
import { Slides } from "./pages/Slides"
import { Layout } from "./components/Layout"

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path=":path/SLIDES.mdx" element={<Slides />} />
            <Route path="/*" element={<Page />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
