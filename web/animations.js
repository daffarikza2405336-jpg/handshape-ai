const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (!reducedMotion) {
  try {
    const [gsapModule, scrollTriggerModule, motion, lenisModule] = await Promise.all([
      import("https://cdn.jsdelivr.net/npm/gsap@3.12.5/+esm"),
      import("https://cdn.jsdelivr.net/npm/gsap@3.12.5/ScrollTrigger/+esm"),
      import("https://cdn.jsdelivr.net/npm/motion@12.6.3/+esm"),
      import("https://cdn.jsdelivr.net/npm/lenis@1.1.20/+esm"),
    ]);

    const gsap = gsapModule.gsap;
    const { ScrollTrigger } = scrollTriggerModule;
    const { animate, hover, inView } = motion;
    const Lenis = lenisModule.default;

    gsap.registerPlugin(ScrollTrigger);

    const lenis = new Lenis({
      duration: 1.1,
      smoothWheel: true,
      anchors: { offset: -86 },
    });

    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    function splitWords(heading) {
      const walker = document.createTreeWalker(heading, NodeFilter.SHOW_TEXT);
      const textNodes = [];
      while (walker.nextNode()) {
        if (walker.currentNode.nodeValue.trim()) textNodes.push(walker.currentNode);
      }

      for (const textNode of textNodes) {
        const fragment = document.createDocumentFragment();
        for (const part of textNode.nodeValue.split(/(\s+)/)) {
          if (/^\s+$/.test(part)) {
            fragment.append(part);
          } else if (part) {
            const word = document.createElement("span");
            word.className = "reveal-word";
            word.textContent = part;
            fragment.append(word);
          }
        }
        textNode.replaceWith(fragment);
      }

      heading.setAttribute("aria-label", heading.innerText.replace(/\s+/g, " ").trim());
      return heading.querySelectorAll(".reveal-word");
    }

    document.querySelectorAll(".hero h1, .section-heading h2, .closing-section h2").forEach((heading) => {
      const words = splitWords(heading);
      gsap.from(words, {
        yPercent: 110,
        opacity: 0,
        rotateX: -16,
        transformOrigin: "center bottom",
        duration: 0.75,
        stagger: 0.09,
        ease: "power3.out",
        scrollTrigger: {
          trigger: heading,
          start: "top 88%",
          once: true,
        },
      });
    });

    document.querySelectorAll(".gesture-card, .process-step, .team-card").forEach((card, index) => {
      inView(card, (element) => {
        animate(element, { opacity: [0, 1], y: [24, 0] }, {
          duration: 0.58,
          delay: (index % 4) * 0.08,
          ease: "easeOut",
        });
      }, { amount: 0.15 });

      hover(card, (element) => {
        animate(element, { y: -5, scale: 1.015 }, { duration: 0.22, ease: "easeOut" });
        return () => animate(element, { y: 0, scale: 1 }, { duration: 0.28, ease: "easeOut" });
      });
    });
  } catch (error) {
    console.error("Could not initialize page animations.", error);
  }
}
