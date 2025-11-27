import Footer from './Footer';
import { Header } from './Header';

type Props = {
  children: React.ReactNode;
};

export function Layout({ children }: Props) {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        {children}
      </main>
      <Footer />
    </div>
  )
}