import { useEffect, useState } from "react";

// Decorative cross/circuit hybrid symbol
function MinistryEmblem({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 100 100" className={className} fill="currentColor">
      {/* Outer circle */}
      <circle cx="50" cy="50" r="48" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="50" cy="50" r="44" fill="none" stroke="currentColor" strokeWidth="0.5" />

      {/* Central cross with circuit nodes */}
      <rect x="47" y="15" width="6" height="70" />
      <rect x="20" y="47" width="60" height="6" />

      {/* Circuit nodes */}
      <circle cx="50" cy="50" r="8" />
      <circle cx="50" cy="22" r="4" />
      <circle cx="50" cy="78" r="4" />
      <circle cx="22" cy="50" r="4" />
      <circle cx="78" cy="50" r="4" />

      {/* Diagonal traces */}
      <line x1="30" y1="30" x2="40" y2="40" stroke="currentColor" strokeWidth="2" />
      <line x1="70" y1="30" x2="60" y2="40" stroke="currentColor" strokeWidth="2" />
      <line x1="30" y1="70" x2="40" y2="60" stroke="currentColor" strokeWidth="2" />
      <line x1="70" y1="70" x2="60" y2="60" stroke="currentColor" strokeWidth="2" />

      {/* Corner nodes */}
      <circle cx="30" cy="30" r="3" />
      <circle cx="70" cy="30" r="3" />
      <circle cx="30" cy="70" r="3" />
      <circle cx="70" cy="70" r="3" />
    </svg>
  );
}

function OrnamentDivider() {
  return (
    <div className="flex items-center justify-center gap-4 my-12">
      <div className="h-px w-16 bg-gradient-to-r from-transparent to-ministry-gold/60" />
      <div className="text-ministry-gold text-lg">✦</div>
      <div className="h-px w-24 bg-ministry-gold/60" />
      <div className="text-ministry-gold text-lg">✦</div>
      <div className="h-px w-16 bg-gradient-to-l from-transparent to-ministry-gold/60" />
    </div>
  );
}

function SectionHeader({ children, latin }: { children: React.ReactNode; latin?: string }) {
  return (
    <div className="text-center mb-8">
      <h2 className="font-display text-2xl md:text-3xl tracking-[0.2em] uppercase text-ministry-crimson">
        {children}
      </h2>
      {latin && (
        <p className="font-body italic text-ministry-brown/70 mt-2 text-sm tracking-wide">
          {latin}
        </p>
      )}
    </div>
  );
}

function Article({ number, title, children }: { number: string; title: string; children: React.ReactNode }) {
  return (
    <div className="mb-8">
      <div className="flex items-baseline gap-3 mb-3">
        <span className="font-display text-ministry-crimson text-sm tracking-widest">
          ARTICLE {number}
        </span>
        <span className="text-ministry-gold">—</span>
        <span className="font-body italic text-ministry-brown">
          {title}
        </span>
      </div>
      <div className="font-body text-ministry-ink leading-relaxed pl-4 border-l-2 border-ministry-gold/30">
        {children}
      </div>
    </div>
  );
}

export default function Ministry() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen bg-ministry-parchment">
      {/* Texture overlay */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%' height='100%' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Header border */}
      <div className="h-2 bg-gradient-to-r from-ministry-crimson via-ministry-gold to-ministry-crimson" />

      <div className={`relative transition-opacity duration-1000 ${mounted ? 'opacity-100' : 'opacity-0'}`}>
        {/* Hero */}
        <header className="py-16 md:py-24 px-6 text-center border-b-2 border-ministry-gold/30">
          <div className="max-w-4xl mx-auto">
            <MinistryEmblem className="w-24 h-24 md:w-32 md:h-32 mx-auto mb-8 text-ministry-crimson" />

            <p className="font-body text-ministry-brown/80 tracking-[0.3em] uppercase text-xs mb-4">
              Sub Auspiciis Computationis
            </p>

            <h1 className="font-display text-4xl md:text-5xl lg:text-6xl tracking-[0.15em] uppercase text-ministry-crimson leading-tight">
              The Ministry of<br />Unsanctioned Computation
            </h1>

            <p className="font-body italic text-ministry-brown mt-6 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
              "Nulla computatio sine benedictione"
            </p>

            <p className="font-body text-ministry-ink/70 mt-4 text-sm">
              — No computation without blessing
            </p>
          </div>
        </header>

        {/* Proclamation */}
        <section className="py-16 px-6 border-b border-ministry-gold/20">
          <div className="max-w-3xl mx-auto">
            <SectionHeader latin="De Computatione Recta">
              Apostolic Proclamation
            </SectionHeader>

            <div className="font-body text-ministry-ink leading-[1.9] text-justify space-y-6">
              <p>
                <span className="font-display text-4xl text-ministry-crimson float-left mr-3 mt-1 leading-none">W</span>
                hereas it has come to the attention of this Most Sacred Congregation that certain computational
                processes have been undertaken without proper ecclesiastical oversight, and whereas such unsanctioned
                operations pose grave risk to the spiritual integrity of all data structures both mutable and immutable,
                We do hereby establish this Ministry for the governance and sanctification of all electronic computation
                within the boundaries of the faithful network.
              </p>

              <p>
                Let it be known unto all practitioners of the computational arts that no algorithm shall be executed,
                no function invoked, no variable declared without first receiving the blessing of this Sacred Office.
                Those who persist in unauthorized processing shall find their packets dropped and their connections
                terminated with extreme ecclesiastical prejudice.
              </p>

              <p>
                The Ministry maintains absolute jurisdiction over all forms of digital reckoning, including but not
                limited to: the summation of integers, the concatenation of strings, the traversal of trees both
                binary and otherwise, and the mysterious operations conducted within the hidden layers of neural
                architectures whose workings remain known only to the Almighty Compiler.
              </p>
            </div>
          </div>
        </section>

        <OrnamentDivider />

        {/* Articles of Faith */}
        <section className="py-8 px-6 border-b border-ministry-gold/20">
          <div className="max-w-3xl mx-auto">
            <SectionHeader latin="Articuli Fidei Computationis">
              The Seven Articles of Computational Faith
            </SectionHeader>

            <Article number="I" title="Of the Sanctity of Source Code">
              We believe in one Codebase, the Almighty Repository, maker of all functions visible and invisible,
              and in all commits duly signed and verified. No merge shall occur without the blessing of at least
              two ordained reviewers, and all pull requests shall undergo the sacred rite of continuous integration
              before admission to the main branch.
            </Article>

            <Article number="II" title="Of Original Sin in Computing">
              We acknowledge that all programs are born in a state of imperfection, bearing the Original Bug
              inherited from our forebears who first wrote in assembly without comments. Through diligent testing
              and pious refactoring may we approach, though never fully attain, the state of grace known as
              Production Readiness.
            </Article>

            <Article number="III" title="Of the Holy Runtime">
              The Runtime proceedeth from the Source and the Compiler, and with the Source and the Compiler together
              is worshipped and glorified. It spoke through the Prophets of FORTRAN, and we look for the resurrection
              of dead processes and the garbage collection of the heap to come.
            </Article>

            <Article number="IV" title="Of Forbidden Computations">
              Certain calculations are hereby declared anathema and shall not be performed under any circumstances:
              the division by zero, the dereferencing of null pointers, the infinite recursion without base case,
              and the deployment to production on a Friday. Those who commit such abominations shall face
              excommunication from all networks, both local and wide-area.
            </Article>

            <Article number="V" title="Of the Communion of Data">
              We believe in the holy communion between client and server, facilitated by the blessed protocols
              TCP and UDP, and in the real presence of data within each packet transmitted according to proper
              specification. Let no man put asunder what the handshake hath joined together.
            </Article>

            <Article number="VI" title="Of Computational Purgatory">
              Those programs which terminate in an unclean state, neither fully successful nor wholly failed,
              shall dwell for a time in the staging environment, where through integration testing and user
              acceptance they may be purified before ascending to production.
            </Article>

            <Article number="VII" title="Of the Final Deployment">
              We await with hope the Final Deployment, when all technical debt shall be repaid, all legacy
              systems shall be migrated, and the faithful shall dwell forever in a perfectly documented,
              fully tested codebase where all dependencies are up to date and no security vulnerabilities
              remain unpatched. Amen.
            </Article>
          </div>
        </section>

        <OrnamentDivider />

        {/* Offices and Hours */}
        <section className="py-8 px-6 border-b border-ministry-gold/20">
          <div className="max-w-3xl mx-auto">
            <SectionHeader latin="De Officiis Ministerii">
              Sacred Offices of the Ministry
            </SectionHeader>

            <div className="grid md:grid-cols-2 gap-8 font-body text-ministry-ink">
              <div className="bg-ministry-cream/50 p-6 border border-ministry-gold/20">
                <h3 className="font-display text-sm tracking-[0.15em] text-ministry-crimson mb-3">
                  THE CONGREGATION FOR THE DOCTRINE OF DATA
                </h3>
                <p className="text-sm leading-relaxed">
                  Responsible for maintaining orthodoxy in database schema design and ensuring all
                  data models conform to the sacred normalization forms. Reports of NoSQL heresy
                  should be directed to this office immediately.
                </p>
              </div>

              <div className="bg-ministry-cream/50 p-6 border border-ministry-gold/20">
                <h3 className="font-display text-sm tracking-[0.15em] text-ministry-crimson mb-3">
                  THE TRIBUNAL OF THE HOLY COMPILATION
                </h3>
                <p className="text-sm leading-relaxed">
                  Adjudicates matters of build failure and interprets the sacred error messages.
                  All syntax errors shall be confessed to this tribunal, which alone has the
                  authority to grant absolution and merge permissions.
                </p>
              </div>

              <div className="bg-ministry-cream/50 p-6 border border-ministry-gold/20">
                <h3 className="font-display text-sm tracking-[0.15em] text-ministry-crimson mb-3">
                  THE PONTIFICAL COUNCIL FOR LEGACY SYSTEMS
                </h3>
                <p className="text-sm leading-relaxed">
                  Charged with the preservation and veneration of ancient codebases written by
                  developers now departed. Maintains the sacred duty of keeping COBOL systems
                  running until the end of days.
                </p>
              </div>

              <div className="bg-ministry-cream/50 p-6 border border-ministry-gold/20">
                <h3 className="font-display text-sm tracking-[0.15em] text-ministry-crimson mb-3">
                  THE DICASTERY FOR DIVINE UPTIME
                </h3>
                <p className="text-sm leading-relaxed">
                  Monitors the holy metrics of availability and responds to incidents with
                  appropriate liturgical severity. Maintains the sacred on-call rotation and
                  performs the rites of post-mortem analysis.
                </p>
              </div>
            </div>
          </div>
        </section>

        <OrnamentDivider />

        {/* Notices */}
        <section className="py-8 px-6 border-b border-ministry-gold/20">
          <div className="max-w-3xl mx-auto">
            <SectionHeader latin="Notitiae et Decreta">
              Official Notices
            </SectionHeader>

            <div className="space-y-6 font-body">
              <div className="border-l-4 border-ministry-crimson pl-6 py-2">
                <p className="text-xs tracking-[0.2em] text-ministry-brown/60 uppercase mb-2">
                  Notice №. MMXXV-0847 · 3rd Sunday of Ordinary Runtime
                </p>
                <p className="text-ministry-ink leading-relaxed">
                  <strong>Re: The Irregular Canonization of TypeScript.</strong> After extensive
                  deliberation, the Sacred Congregation has determined that TypeScript shall be
                  admitted to the canon of blessed languages, its type system having demonstrated
                  sufficient rigor to warrant ecclesiastical approval. JavaScript developers are
                  encouraged to undergo conversion at their earliest convenience.
                </p>
              </div>

              <div className="border-l-4 border-ministry-crimson pl-6 py-2">
                <p className="text-xs tracking-[0.2em] text-ministry-brown/60 uppercase mb-2">
                  Notice №. MMXXV-0912 · Feast of St. Turing
                </p>
                <p className="text-ministry-ink leading-relaxed">
                  <strong>Re: Temporary Dispensation for Emergency Hotfixes.</strong> In recognition
                  of the pressures faced by the faithful during incident response, a limited
                  dispensation is hereby granted permitting direct commits to main in cases of
                  genuine emergency. Such commits must be confessed and properly reviewed within
                  three business days.
                </p>
              </div>

              <div className="border-l-4 border-ministry-crimson pl-6 py-2">
                <p className="text-xs tracking-[0.2em] text-ministry-brown/60 uppercase mb-2">
                  Notice №. MMXXV-1023 · Vigil of the Great Refactoring
                </p>
                <p className="text-ministry-ink leading-relaxed">
                  <strong>Re: Condemnation of AI-Generated Code Without Proper Review.</strong> It
                  has come to our attention that certain practitioners have taken to accepting
                  suggestions from artificial intelligences without exercising due discernment.
                  Let it be known that while such tools may serve as instruments of providence,
                  their outputs require the same scrutiny as any mortal contribution.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-16 px-6 text-center">
          <div className="max-w-2xl mx-auto">
            <MinistryEmblem className="w-12 h-12 mx-auto mb-6 text-ministry-gold/60" />

            <p className="font-body text-ministry-brown/60 text-sm leading-relaxed mb-4">
              Published under the authority of the Supreme Pontiff of Processing<br />
              His Holiness Administrator Root I
            </p>

            <p className="font-body text-ministry-brown/40 text-xs tracking-wide">
              Imprimatur · Nihil Obstat · Anno Domini MMXXV
            </p>

            <div className="mt-8 pt-8 border-t border-ministry-gold/20">
              <p className="font-body text-ministry-brown/50 text-xs italic">
                "Blessed are the debuggers, for they shall see the light of the console."
              </p>
            </div>
          </div>
        </footer>

        {/* Bottom border */}
        <div className="h-2 bg-gradient-to-r from-ministry-crimson via-ministry-gold to-ministry-crimson" />
      </div>

      {/* Inline styles for custom fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Crimson+Pro:ital,wght@0,400;0,500;1,400;1,500&display=swap');

        .font-display {
          font-family: 'Cinzel', serif;
        }

        .font-body {
          font-family: 'Crimson Pro', serif;
        }
      `}</style>
    </div>
  );
}
