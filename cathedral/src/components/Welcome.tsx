
export function Welcome() {
  return (
    <div className="text-muted-foreground">
      <pre 
        className="not-prose text-xs md:text-sm rounded"
      >
        {/* Rubifont on figlet */}
{`
▗▄▄▖▗▄▄▄▖▗▖  ▗▖ ▗▄▄▖▗▖    ▗▄▖ ▗▄▄▖ 
▐▌ ▐▌ █  ▐▛▚▖▐▌▐▌   ▐▌   ▐▌ ▐▌▐▌ ▐▌
▐▛▀▘  █  ▐▌ ▝▜▌▐▌▝▜▌▐▌   ▐▛▀▜▌▐▛▀▚▖
▐▌  ▗▄█▄▖▐▌  ▐▌▝▚▄▞▘▐▙▄▄▖▐▌ ▐▌▐▙▄▞▘                                                                    
`}
      </pre>
      <div className="text-xs mt-2 font-mono">
        PingLab is a repository for running experiments on PING Spiking Neural Networks.
      </div>
    </div>
  )
}